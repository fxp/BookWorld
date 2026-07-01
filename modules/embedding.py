import sys
sys.path.append("../")
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction
import os

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class EmbeddingModel(EmbeddingFunction[Documents]):
    """Local HuggingFace/ModelScope embedding model (bge, luotuo, ...).
    The heavy `modelscope`/`torch` imports are loaded lazily so that the
    API-embedding path does not require them to be installed."""

    def __init__(self, model_name, language='en'):
        from modelscope import AutoModel, AutoTokenizer  # lazy: avoids torch unless used
        import torch  # noqa: F401
        from bw_utils import get_child_folders
        self._torch = torch
        self.model_name = model_name
        self.language = language
        cache_dir = "~/.cache/modelscope/hub"
        model_provider = model_name.split("/")[0]
        model_smallname = model_name.split("/")[1]
        model_path = os.path.join(cache_dir, f"models--{model_provider}--{model_smallname}/snapshots/")

        if os.path.exists(model_path) and get_child_folders(model_path):
            try:
                model_path = os.path.join(model_path, get_child_folders(model_path)[0])
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
            except Exception as e:
                print(e)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, input):
        inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        return embeddings


class OpenAIEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1/embeddings", api_key_field="OPENAI_API_KEY"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ[api_key_field]
        )
        self.model_name = model_name

    def __call__(self, input):
        if isinstance(input, str):
            input = input.replace("\n", " ")
            return self.client.embeddings.create(input=[input], model=self.model_name).data[0].embedding
        elif isinstance(input, list):
            return [self.client.embeddings.create(input=[sentence.replace("\n", " ")], model=self.model_name).data[0].embedding for sentence in input]


class ZhipuEmbeddingFunction(EmbeddingFunction[Documents]):
    """BigModel / Zhipu (智谱) embedding API (embedding-3 / embedding-2),
    accessed through the OpenAI-compatible endpoint. Strong on Chinese,
    so it is a good fit for classical-Chinese corpora like 《红楼梦》."""

    BATCH = 64  # Zhipu accepts up to 64 inputs per request

    def __init__(self, model_name="embedding-3", api_key_field="ZHIPUAI_API_KEY", dimensions=None):
        from openai import OpenAI
        api_key = os.getenv(api_key_field) or os.getenv("BIGMODEL_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=ZHIPU_BASE_URL)
        self.model_name = model_name
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        out: Embeddings = []
        for i in range(0, len(input), self.BATCH):
            batch = [t.replace("\n", " ") for t in input[i:i + self.BATCH]]
            kwargs = {"model": self.model_name, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            out.extend([d.embedding for d in resp.data])
        return out

    @staticmethod
    def name() -> str:
        return "zhipu"

    def get_config(self):
        return {"model_name": self.model_name, "dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config):
        return ZhipuEmbeddingFunction(
            model_name=config.get("model_name", "embedding-3"),
            dimensions=config.get("dimensions"),
        )


def get_embedding_model(embed_name, language='en'):
    local_model_dict = {
        "bge-m3": "BAAI/bge-m3",
        "bge-large": f"BAAI/bge-large-{language}",
        "luotuo": "silk-road/luotuo-bert-medium",
        "bert": "google-bert/bert-base-multilingual-cased",
        "bge-small": f"BAAI/bge-small-{language}",
    }
    # Online API embeddings, keyed by the name used in config.json.
    zhipu_names = {"zhipu", "embedding-3", "embedding-2"}
    if embed_name in zhipu_names:
        model_name = "embedding-3" if embed_name in ("zhipu", "embedding-3") else embed_name
        return ZhipuEmbeddingFunction(model_name=model_name)
    online_model_dict = {
        "openai":
            {"model_name": "text-embedding-ada-002",
             "url": "https://api.openai.com/v1/embeddings",
             "api_key_field": "OPENAI_API_KEY"},
    }
    if embed_name in local_model_dict:
        model_name = local_model_dict[embed_name]
        return EmbeddingModel(model_name, language=language)
    if embed_name in online_model_dict:
        model_name = online_model_dict[embed_name]["model_name"]
        api_key_field = online_model_dict[embed_name]["api_key_field"]
        base_url = online_model_dict[embed_name]["url"]
        return OpenAIEmbedding(model_name=model_name, base_url=base_url, api_key_field=api_key_field)
    return EmbeddingModel(embed_name, language=language)
