# LAP

- !! 이 모듈은 단일 쓰레드 환경에서만 안전하게 측정 가능하다 !!
- 실행시간 측정을 위해 코드의 각 부분을 lap하여 lap까지 걸린 실행시간을 저장하는 로그 모듈.
- 연속적으로 실행되는 코드를 측정하는 warterfall lap과 루프 내에서 동작하는 코드를 측정하는 loop lap이 있다.
- waterfall lap은 각 lap을 나타내는 메시지와 이전 lap으로 부터 걸린 시간(elapsed time)을 측정한다.
- loop lap의 경우 각 lap의 메시지와 lap이 실행된 횟수(iteration), 첫 번째 루프에서 걸린 시간(first), 마지막 루프에서 걸린 시간(last), 모든 루프에서의 평균시간(average), 모든 루프에서 가장 긴 시간(max), 모든 루프에서 가장 짧은시간(min)을 저장한다.
