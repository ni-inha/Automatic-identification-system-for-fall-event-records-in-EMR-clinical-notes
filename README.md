# Automatic identification system for fall event records in EMR clinical notes

미세조정 [KLUE/BERT-base](https://huggingface.co/klue/bert-base) 모델을 활용한 EMR 임상노트 내 낙상사건 자동식별 시스템

## Environments

- Hardware
  - GPU (graphics processing unit)
- Software
  - Python 3.8
  - tensorflow==2.10.0
  - keras==2.10.0
  - torch
  - transformers>=4.0.0
  - pandas
  - pymysql
  - cryptography
  - apscheduler
  - tzlocal

## Preparation & Installation

- KLUE_BERT_model_weights.zip (1.5GB)
  - [One Drive](https://o365inha-my.sharepoint.com/:u:/g/personal/time_office_inha_ac_kr/EbfaEhre8KJNiEsPJiNR1ZABXXUyhnjL_X0rQt2WEkaqzA?e=497yv5)
  - [Google Drive](https://drive.google.com/file/d/10XGR6fHmS_wVfQwAQ0CX87i_xABfb-Go/view?usp=sharing)

위 압축파일을 프로젝트 폴더에 다운받고, 아래 명령어들을 실행합니다.

```bash
$ unzip KLUE_BERT_model_weights.zip
$ pip install -r requirements.txt
```

## Usage

- 프로그램과 연동할 데이터베이스의 기본 정보를 변경합니다. (191~195번 줄)
  - ```py
      host_name = "127.0.0.1"
      host_port = 3306
      username = "root"
      password = "P@$$W0RD"
      database_name = "neticaj_fall"
    ```
- 프로그램의 스케줄 주기를 변경합니다. 기본 설정 값은 00시 05분 입니다. (189번 줄)
  - ```python
      @sched.scheduled_job('cron', hour='00', minute='05', id='predict_from_db')
      def job():
          ......
    ```
