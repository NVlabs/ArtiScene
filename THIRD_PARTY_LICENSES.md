# Third Party Licenses

This file contains the licenses for all third-party open source software used in ArtiScene.

## Core Dependencies

### PyTorch Ecosystem
- **torch==1.13.1** - BSD-3-Clause License
- **torchvision==0.14.1** - BSD-3-Clause License  
- **torchmetrics==0.6.0** - Apache License 2.0
- **pytorch-lightning==1.4.2** - Apache License 2.0

### Computer Vision & AI Libraries
- **opencv-python==4.6.0.66** - Apache License 2.0
- **opencv-python-headless==4.11.0.86** - Apache License 2.0
- **Pillow==9.5.0** - Historical Permission Notice and Disclaimer (HPND)
- **albumentations==1.3.0** - MIT License
- **segment_anything==1.0** - Apache License 2.0
- **scikit-image** - BSD-3-Clause License
- **scikit-learn** - BSD-3-Clause License
- **supervision==0.25.1** - MIT License

### Transformers & NLP
- **transformers==4.48.0** - Apache License 2.0
- **huggingface-hub==0.27.1** - Apache License 2.0  
- **tokenizers==0.21.0** - Apache License 2.0
- **timm==0.9.2** - Apache License 2.0

### Scientific Computing
- **numpy==1.23.0** - BSD-3-Clause License
- **scipy** - BSD-3-Clause License
- **pandas==2.2.3** - BSD-3-Clause License
- **matplotlib==3.10.0** - PSF-based License
- **imageio==2.37.0** - BSD-2-Clause License

### Deep Learning Infrastructure  
- **einops==0.3.0** - MIT License
- **kornia==0.5.0** - Apache License 2.0
- **xformers==0.0.16** - BSD-3-Clause License
- **open-clip-torch==2.0.2** - MIT License

### Visualization & Graphics
- **Brotli==1.1.0** - MIT License
- **bpy==4.0.0** - GPL-3.0 License
- **OpenEXR==3.3.2** - BSD-3-Clause License

### API & Web Services
- **openai==1.60.2** - MIT License
- **requests==2.32.3** - Apache License 2.0
- **urllib3==2.3.0** - MIT License
- **aiohttp==3.11.11** - Apache License 2.0
- **httpx==0.28.1** - BSD-3-Clause License

### Configuration & Templates
- **Jinja2==3.1.5** - BSD-3-Clause License
- **PyYAML==6.0.2** - MIT License
- **omegaconf==2.1.1** - BSD-3-Clause License
- **hydra-core==1.1.1** - Apache License 2.0

### Development & Utilities
- **tqdm==4.67.1** - MIT License/Mozilla Public License 2.0
- **click==8.1.8** - BSD-3-Clause License
- **rich==13.9.4** - MIT License
- **packaging==24.2** - Apache License 2.0/BSD-2-Clause License

### ML Operations & Tracking
- **wandb==0.19.4** - MIT License
- **tensorboard==2.18.0** - Apache License 2.0

### Data Processing
- **pycocotools==2.0.8** - BSD-2-Clause License
- **faiss-cpu** - MIT License
- **boto3==1.36.7** - Apache License 2.0
- **botocore==1.36.7** - Apache License 2.0

## License Texts

### Apache License 2.0
Used by: torch, torchmetrics, pytorch-lightning, opencv-python, segment_anything, transformers, huggingface-hub, tokenizers, timm, kornia, requests, hydra-core, tensorboard, boto3, botocore, and others.

Full license text available at: https://www.apache.org/licenses/LICENSE-2.0

### MIT License
Used by: albumentations, supervision, einops, open-clip-torch, Brotli, openai, urllib3, PyYAML, tqdm, rich, wandb, faiss-cpu, and others.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BSD-3-Clause License
Used by: torch, torchvision, scikit-image, scikit-learn, numpy, scipy, xformers, OpenEXR, httpx, Jinja2, click, and others.

```
BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### BSD-2-Clause License  
Used by: imageio, pycocotools, and others.

```
BSD 2-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### GPL-3.0 License
Used by: bpy==4.0.0 (Blender Python API)

**Note**: This dependency includes GPL-3.0 licensed code. Users should be aware of the copyleft requirements when distributing derivative works.

Full license text available at: https://www.gnu.org/licenses/gpl-3.0.html

## External Tools & Dependencies

This project also relies on several external tools and repositories that should be installed separately:

### GroundingDINO
- Repository: https://github.com/IDEA-Research/GroundingDINO
- License: Apache License 2.0

### MaskFormer/Mask2Former
- Repository: https://github.com/facebookresearch/MaskFormer  
- License: CC-BY-NC 4.0 (Non-commercial license)

### ODISE
- Repository: https://github.com/NVlabs/ODISE
- License: NVIDIA Source Code License

### Inpaint-Anything
- Repository: https://github.com/geekyutao/Inpaint-Anything
- License: Apache License 2.0

### Pix2Gestalt
- Repository: https://github.com/cvlab-columbia/pix2gestalt
- License: MIT License

### SD-DINO
- Repository: https://github.com/Junyi42/sd-dino
- License: Apache License 2.0

## Notice

This software includes components from various open source projects. Each component's license terms are preserved and must be followed when redistributing or modifying this software.

For complete and up-to-date license information, please refer to the individual package documentation and source repositories. 