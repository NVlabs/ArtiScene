# -----------------------------------------------------------------------------
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#
# Code for building NVIDIA proprietary Edify APIs.
# It cannot be released to the public in any form.
# If you want to use the code for other NVIDIA proprietary products,
# please contact the Deep Imagination Research Team (dir@exchange.nvidia.com).
# -----------------------------------------------------------------------------

""" Script to deploy the NGC Container as a NVIDIA Cloud Function. """


import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from pathlib import Path

logger = logging.getLogger("deploy")

DEFAULT_TIMEOUT = 300  # 60 seconds

# TODO: Extend this to cover all types of extensions we support.
ASSET_TYPE_MAP = {
    ".jpg": {"content_type": "image/jpeg", "data_type": "BYTES", "data_shape": [1]},
    ".png": {"content_type": "image/png", "data_type": "BYTES", "data_shape": [1]},
    ".glb": {"content_type": "binary/octet-stream", "data_type": "BYTES", "data_shape": [1]},
}


class NVCFException(Exception):
    """Class for raising an NVCF exception."""

@dataclass
class NVCFAsset:
    """Dataclass representing an asset"""

    url: str
    name: Optional[str] = None
    description: Optional[str] = None
    asset_id: Optional[str] = None
    content_type: Optional[str] = None
    local_filepath: Optional[str] = None
    data_type: Optional[str] = None
    data_shape: Optional[List[int]] = None

    def upload(self):
        """Upload an asset to NVCF."""
        if not self.local_filepath or not os.path.exists(self.local_filepath):
            raise FileNotFoundError(f"{self.local_filepath} not found")

        with open(self.local_filepath, "rb") as file:
            file_binary = file.read()
            headers = {
                "Content-Type": self.content_type,
                "x-amz-meta-nvcf-asset-description": self.description,
            }
            resp = requests.put(
                self.url,
                data=file_binary,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code != 200:
                raise NVCFException(
                    f"Failed to upload asset : {asdict(self)} "
                    f"    Status code: {resp.status_code}."
                    f"    Content: {_dump(resp)}."
                )
            return resp

    def download(self, filepath: Optional[str] = None) -> None:
        """Download an asset to the given directory from the signed asset URL."""
        resp = requests.get(self.url, timeout=DEFAULT_TIMEOUT)

        if resp.status_code != 200:
            raise NVCFException(
                f"Failed to download: {self.url} "
                f"    Status code: {resp.status_code}."
                f"    Content: {_dump(resp)}."
            )
        if not filepath:
            filepath = self.local_filepath
        if not filepath:
            raise NVCFException("No local filepath provided")
        with open(filepath, "wb") as file:
            file.write(resp.content)


@dataclass
class NVCF:
    """NVCF Base Class."""

    nvcf_api: str
    nvcf_token: str
    api_version: str = "v2"
    _scope_joiner: str = "%20"
    _grant_type: str = "client_credentials"

    @property
    def auth_header(self) -> dict:
        """Returns the Authentication header."""
        return {"Authorization": f"Bearer {self.nvcf_token}"}

    def invoke_function_version(self, function_id: str, version_id: str, command: str, assets: list[NVCFAsset]) -> None:
        """Run Inference with an NVCF function."""
        body = {
            "requestHeader": {
                "inputAssetReferences": [asset.asset_id for asset in assets],
            },
            "requestBody": {
                "inputs": [
                    {
                        "name": "command",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [command],
                    }
                ]
                + [
                    {
                        "name": asset.name,
                        "shape": asset.data_shape,
                        "datatype": asset.data_type,
                        "data": [asset.asset_id],
                    }
                    for asset in assets
                ]
            },
        }

        if version_id:
            url=f"\n{self.nvcf_api}/{self.api_version}/nvcf/exec/functions/{function_id}/versions/{version_id}"
        else:
            url=f"\n{self.nvcf_api}/{self.api_version}/nvcf/exec/functions/{function_id}"
        resp = requests.post(
            url=url,
            headers=self.auth_header,
            json=body,
            timeout=DEFAULT_TIMEOUT,
        )

        if resp.status_code != 200 and resp.status_code != 202:
            raise NVCFException(
                f"Unable to check details of Inference Request:"
                f"    Status code: {resp.status_code}."
                f"    Content: {_dump(resp)}."
            )
        return (resp.json(), resp.status_code)

    def create_asset(self, filepath: str, name: str = None, description: str = None) -> NVCFAsset:
        """Upload an asset to NVCF."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        filename, extension = os.path.splitext(filepath)
        short_filename = filename.split("/")[-1]
        asset_type = ASSET_TYPE_MAP.get(extension, "UNKNOWN")
        assert asset_type != "UNKNOWN"

        headers = self.auth_header
        headers.update({"Content-Type": "application/json", "accept": "application/json"})
        description = description if description else f"{short_filename}{extension}"
        body = {
            "contentType": asset_type["content_type"],
            "description": description,
        }
        resp = requests.post(
            url=f"\n{self.nvcf_api}/{self.api_version}/nvcf/assets",
            headers=headers,
            json=body,
            timeout=DEFAULT_TIMEOUT,
        )

        if resp.status_code != 200:
            raise NVCFException(
                f"Failed to retrieve asset_id or url"
                f"    Status code: {resp.status_code}."
                f"    Content: {_dump(resp)}."
            )

        name = name if name else short_filename
        res_json = resp.json()
        return NVCFAsset(
            url=res_json["uploadUrl"],
            name=name,
            description=description,
            asset_id=res_json["assetId"],
            content_type=asset_type["content_type"],
            data_type=asset_type["data_type"],
            data_shape=asset_type["data_shape"],
            local_filepath=filepath,
        )

    def poll_until_done(self, req_id: str, wait_seconds: float = 1.0, retries: int = 100) -> Optional[Dict]:
        """Poll a request."""
        for _ in range(retries):
            resp = self.get_request_status(req_id)
            status_code = resp["status_code"]
            if status_code == 200:
                return resp
            elif status_code == 202:
                logging.info(f"Continue polling request {req_id}")
                print((f"Continue polling request {req_id}"))
            else:
                raise NVCFException(f"Error {status_code}")
            time.sleep(wait_seconds)

    def get_request_status(self, req_id: str) -> None:
        """Get details from an NVCF inference call."""
        resp = requests.get(
            url=f"{self.nvcf_api}/{self.api_version}/nvcf/exec/status/{req_id}",
            headers=self.auth_header,
            timeout=DEFAULT_TIMEOUT,
        )
        json = resp.json()
        json["status_code"] = resp.status_code
        return json


def _dump(resp):
    return json.dumps(resp.json(), indent=4) if resp.content else None

from typing import NamedTuple, Optional, List, Dict, Literal
from dataclasses import dataclass
import requests
import json

@dataclass
class FunctionCall:
    function: 'Function'
    client: NVCF
    command: str
    inputs: List[NVCFAsset]
    output: Optional[NVCFAsset] = None
    response: Optional[Dict] = None

    def __call__(self):
        response, _ = self.client.invoke_function_version(self.function.id, self.function.version, self.command, self.inputs)
        self.response = response
        url = response.get('responseReference', None)
        if url: self.output = NVCFAsset(url)

    def add_input(self, filepath: str):
        asset = self.client.create_asset(filepath=filepath, name=f'image_assetId_{len(self.inputs)}')
        asset.upload()
        self.inputs.append(asset)
    
    def poll(self):
        status = self.response.get('status', None)
        if status != 'fulfilled':
            request_id = self.response['reqId']
            self.response = self.client.poll_until_done(request_id)
        url = self.response.get('responseReference', None)
        if url: self.output = NVCFAsset(url)

class Function(NamedTuple):
    id: str
    token: str
    version: str = None

    def __call__(self, command, inputs=None, output='output.zip'):
        if inputs is None: inputs = []
        call = self.init_call(command, inputs)
        call()
        call.poll()
        call.output.download(output)

    def init_call(self, command, inputs):
        client = NVCF(nvcf_api="https://api.nvcf.nvidia.com", nvcf_token=self.token)
        call = FunctionCall(self, client, command, [])
        for input in inputs:
            call.add_input(input)
        return call

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='the structure should be root_dir/folder1, folder2, ..., where each folder if of a scene')
    parser.add_argument('--folders', type=str, help='the folders you want to generate wallpapers for. if for all folders under the root_dir, use "all", otherwise separate the target folders by a comma, no space')
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--start_ind", type=int, default=0, help='when you have multiple folders, the start index of running this script.')
    parser.add_argument("--end_ind", type=int, default=1, help='when you have multiple folders, the end index of running this script.')
    return parser.parse_args()

if __name__ == "__main__":
    start2=time.time()
    args = parse_args()
    # this calls the picasso-alpha function
    function = Function(id='35a04cdd-4ea6-4769-b477-d510787442b3',\
                        version='f6157832-850a-4f3f-8929-9bc9ef24566d',
                        token='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')
    

    d=args.root_dir
    retry_delay=3
    """
    Processes each folder in directory `d` using function `f`.
    If a folder fails, retries until all folders are processed.

    :param d: Directory containing the folders to process
    :param retry_delay: Delay in seconds between retries
    """
    # Get a list of all folders in directory `d`
    folders=args.folders.split(',')
    all_folders = sorted([os.path.join(d, folder) for folder in folders if os.path.isdir(os.path.join(d, folder))])[args.start_ind:args.end_ind]
    # all_folders = sorted([os.path.join(d, folder) for folder in os.listdir(d) if os.path.isdir(os.path.join(d, folder))])[args.start_ind:args.end_ind]

    # Process folders and handle failures
    while all_folders:
        remaining_folders = []  # Keep track of folders that need to be retried

        for folder in all_folders:
            print(folder)
            args.json_path=os.path.join(folder,'for_image_text23d/furnitures_merged_2_pix2gestalt/furnitures_cropped_seg_rgb_together_merged_2.json')
            args.image_folder=os.path.join(folder,'for_image_text23d/furnitures_merged_2_pix2gestalt/images')
            try:
                ##################
                with open(Path(args.json_path)) as f:
                    objectname2description = json.load(f)
                if 'repeat.json' in os.listdir(os.path.join(folder,'for_image_text23d/furnitures_merged_2_pix2gestalt')):
                    with open(os.path.join(folder,'for_image_text23d/furnitures_merged_2_pix2gestalt/repeat.json')) as f:
                        repeat = json.load(f)
                else:
                    repeat={k:None for k in objectname2description}
                save_root=os.path.dirname(args.json_path)
                save_folder=os.path.join(save_root, 'edify3D')
                os.makedirs(save_folder, exist_ok=True)
                for (k,v) in objectname2description.items():
                    if k+'.zip' in os.listdir(save_folder): continue
                    if k in os.listdir(save_folder): continue
                    if k+'.png' not in os.listdir(args.image_folder): continue
                    if k in repeat and repeat[k] is not None:
                        print('skip repeats!', k, repeat[k])
                        continue
                    if k not in repeat:
                        print('[warning] k not in repeat!!!', k, repeat)
                    print(k)
                    prompt=v['caption']
                    category =k.split('_')[1]
                    if prompt=='':
                        prompt = 'a '+k.split('_')[-1]
                    if category.lower() in ['desk','table']:
                        prompt+=' It has 4 legs.'
                    subdir_name_for_i=k
                    image_path=os.path.join(args.image_folder, k+'.png')
                    print(image_path, prompt, os.path.isfile(image_path))
                    start_time=time.time()
                    # example how to call image23d. image2preview is similar.
                    function('image23d --mesh_target_num_faces=5000 --num_samples=1 --prompt="{p}" --input_image=image_0'.format(p=prompt), inputs=[image_path], output=os.path.join(save_folder, k+'.zip'))
                    print('one function call', time.time()-start_time)
                ##################
            except Exception as e:
                print(f"Failed to process {folder}. Error: {e}")
                remaining_folders.insert(0, folder)  # Add failed folder to retry list
        
        if remaining_folders:
            print(f"Retrying {len(remaining_folders)} folders in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("All folders processed successfully.")
        
        all_folders = remaining_folders  # Update the list to only include failed folders
    time2=time.time()-start2
    print('time2', time2)
    with open(os.path.join(folder,'step6_gen3d_furnitures_time.txt'), 'w') as f:
        f.write(str(time2))
    
