# Dependencies
- python3
- python3-venv
- python3-pip
- cuda toolkit >= 10.1

# Build
## ***주의하세요*** encoded_data가 이미 있다면 되도록 kmeans 디렉토리에서 make 하세요.
이 프로젝트를 빌드하면 다음 과정을 거치게 되며 1번은 시간이 오래걸립니다.
1. python virtual environment를 생성하여 numpy, keras, tensorflow등의 라이브러리를 설치하고, autoencoder를 실행합니다.
2. sequential_kmeans 와 parallel_kmeans를 차례로 빌드하여 실행합니다. 이 때 1번의 결과로 만들어진 encoded_mnist 데이터가 사용됩니다.

# Project sources
- kmeans/sequential 에 c++로 작성된 sequential 버전 kmeans 알고리즘이 구현되어 있습니다.
- kmeans/parallel 에 cuda c++로 작성된 parallel 버전 kmeans 알고리즘이 구현되어 있습니다.
- kmeans/include/config.hh 에서 다음을 설정하실 수 있습니다.
```c++
#define DATA_SCALE 1  // 주어진 data를 해당 상수만큼 복사합니다.
#define THREASHOLD 50 // 수렴 주기의 최대 값을 정합니다.
```
- time log를 위한 모듈은 따로 제공되지 않습니다. 육안으로도 명확히 확인할 수 있으니까요..