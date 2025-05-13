from custom_node_helper import CustomNodeHelper

class PuLID_ComfyUI(CustomNodeHelper):
    @staticmethod
    def weights_map(base_url):
        return {
            "ip-adapter_pulid_sdxl_fp16.safetensors": {
                "url": "https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors",
                "dest": "ComfyUI/models/pulid/",
            },
            "parsing_parsenet.pth": {
                "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
                "dest": "ComfyUI/models/facexlib/",
            },
        }

    @staticmethod
    def add_weights(weights_to_download, node):
        if node.is_type_in([
            "PulidLoader",
            "PulidApply",
        ]):
            weights_to_download.append("ip-adapter_pulid_sdxl_fp16.safetensors")
            weights_to_download.append("models/antelopev2")
            weights_to_download.append("parsing_parsenet.pth")
            weights_to_download.append("pony_realism_23.safetensors")