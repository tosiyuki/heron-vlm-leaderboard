import pytest
#from adapters.base_adapter import BaseAdapter
from plugins.llava_jp_adapter import LlavaJpAdapter

@pytest.mark.asyncio
async def test_llava_jp_adapter():
    adapter = LlavaJpAdapter("toshi456/llava-jp-1.3b-v1.1", "cuda", {"max_length": 50, "temperature": 0.7})
    assert adapter.model_name == "toshi456/llava-jp-1.3b-v1.1"
    assert await adapter.verify()

    response = await adapter.generate_response("この画像には何が写っていますか？", "test.jpg")
    assert isinstance(response, str)
    assert len(response) > 0

# Add more tests as needed