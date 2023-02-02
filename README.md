# :shirt: Virtual Try-On

CV-10
## 조물주 김영민


## :notebook: Project Abstract


### :page_with_curl: Background
국내 온라인 패션 시장이 코로나19의 영향으로 빠르게 성장했고, 앞으로도 그런 기조를 보일 예정입니다. 그러나 온라인에서 옷을 직접 입어보지 못해 고객들이 자신과 옷이 어울리는지 확인하기 어렵습니다. 저희는 Virtual Try-On 모델을 활용해 온라인 가상 피팅 서비스를 제공하여 많은 고객에게 편리한 서비스를 제공하려고 합니다.
<br/><br/>


### :raising_hand: Members
|   김원회   |   김의진   |  김진섭   |  전영주   |  함수민  |
| :--------: | :--------: | :------: | :-------: | :-------: |
|   T4042   |   T4046   |  T4053   |  T4189   |  T4227  |


<br/>

#### :computer: Tech stack

 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white"/>
<br/>

#### :minidisc: Feature
사용자가 원하는 옷으로 변경 

<br/>
<br/>

#### :low_brightness: 기대효과
- 기존의 가상 피팅 서비스는 실제 인물이 아닌 가상 인물에 제품을 피팅하는 형태가 많음


- 본 프로젝트는 실제 사용자가 원하는 옷을 피팅할 수 있는 형태로 구성하고자 함
<br/>
<br/>


## Data


### :black_square_button: Data resource
![](/imgs/aihub.png)
- AI Hub 패션상품 및 착용 영상 데이터셋
    - 이 중, 정면에서 촬영된 모델과 모델이 입고 있는 상의를 pair로 데이터셋을 구성했습니다.
    - train data: 971장 / test data: 489장
    - 해상도 : 1024 X 768
<br/>

### :black_square_button: Data Preprocessing
HR-VITON의 input으로는 model, cloth images를 바탕으로 전처리 된 입력들이 추가로 필요합니다.


<br/>

#### :small_red_triangle_down: Pose Estimation
![](/imgs/openpose_ex.png)
<br/>

#### :small_red_triangle_down: Human Parsing
![](/imgs/humanparse_ex.png)
<br/>

#### :small_red_triangle_down: Dense Pose
![](/imgs/densepose_ex.png)
<br/>

#### :small_red_triangle_down: Cloth Mask
![](/imgs/clothmask_ex.png)
<br/>

---
## Model Architecture
---

#### Condition Generator
![](/imgs/Conditon_Generator.png)


#### Image Generator
![](/imgs/Image_Generator.png)

<br/>
<br/>



## Service Architecture
![](/imgs/service_arc.png)
<br/>
<br/>


## Demo

<br/>
<br/>

## Reference

:small_red_triangle_down: HR-VITON

Paper: https://arxiv.org/abs/2206.14180
Project page: https://koo616.github.io/HR-VITON


<br/>
<br/>
