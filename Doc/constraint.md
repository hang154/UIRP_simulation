# Constraint 

## 1. global file + scene file 전송 완료시 , 연산 시작 가능 
## 2. 전송 속도는 tras/rec bandwidth 최소값 
## 3. (task.global_file_size + scene 개수 * (각각 scene 의 filesize))  / min(provider.bandwidth, task.bandwidth) = 영상 전송에 걸리는 시간
## 4. GPU는 1번에 1개의 작업만 할당 가능
## 5. 시간 * provider.price_per_gpu_hour = 전체 소모비용


# Objective

## 수요자 : 하나의 작업에 대한 예산, 총 소요시간 최소화
## 공급자 : GPU 연산능력 당 시간당 수익의 최대화

hour * 

## Objective = 