This is DeepLearning Project repo

# 데모 페이지 실행
streamlit run mainpage.py 
mainpage.py에서 데이터셋에 따라 weight 바꿔서 실행

# train (House)
python train.py —img 320 —weights yolov5x.pt —project ./result/FreshTrain/House —data ./dataset/NIa_yolo/House/data.yaml —cache —batch-size -1 —device 0 —epochs 300
다른 데이터셋의 경우, data 경로 바꿔서 실행. dataset/NIa_yolo/Tree/data.yaml 

# 배포 url 
https://deeplearning-project-ar2vfgvflcymz7fzb6uuvg.streamlit.app/
