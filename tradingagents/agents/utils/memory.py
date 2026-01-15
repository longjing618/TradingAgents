import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class FinancialSituationMemory:
    def __init__(self, name, config):
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        https://share.google/aimode/L9TZBAL38BSQ2N7bF
        Truncates a string to have at most `max_tokens` according to the specified encoding.

        Args:
            text: The original string to truncate.
            model_encoding: The name of the tiktoken encoding (e.g., 'cl100k_base', 'p50k_base').
            max_tokens: The maximum number of tokens allowed.

        Returns:
            The truncated string.
        """
        try:
            # Get the appropriate encoding
            # https://share.google/aimode/wlxGmTvtWu6rYKYsr
            encoding = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            # Fallback if encoding name isn't found directly (e.g., model name provided)
            encoding = tiktoken.encoding_for_model(self.embedding)

        # Encode the text into a list of tokens (integers)
        tokens = encoding.encode(text)

        # Slice the list of tokens to the maximum allowed length
        truncated_tokens = tokens[:max_tokens]

        # Decode the truncated list of tokens back into a string
        truncated_text = encoding.decode(truncated_tokens)

        return truncated_text

    def get_embedding_impl_nomic(self, text):
        # The hf nomic-embed-text model supports 8192 context length
        # https://github.com/ollama/ollama/issues/7741#issuecomment-2485808342
        # So git clone the nomic-ai/nomic-embed-text-v1.5 model and load it instead of using ollama
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
        model.eval()

        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        matryoshka_dim = 512

        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].tolist()

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        
        # https://platform.openai.com/docs/api-reference/embeddings
        # There is a max token length limit
        text = self.truncate_text_by_tokens(text=text, max_tokens=1)

        if self.embedding == "nomic-embed-text":
            return self.get_embedding_impl_nomic(text)

        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
