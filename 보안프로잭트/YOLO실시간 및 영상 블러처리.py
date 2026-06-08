import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime  # 시간 라이브러리 추가

# 1. 고속 YOLOv8 나노 세그멘테이션 모델 로드
model = YOLO('yolov8n-seg.pt')

# -------------------------------------------------------------
# [캠 입력 설정] 
cap = cv2.VideoCapture(0)
# -------------------------------------------------------------

if not cap.isOpened():
    print("노트북 카메라를 열 수 없습니다 장치 연결을 확인하세요")
    exit()

# 캠의 해상도(가로, 세로) 및 FPS 가져오기
origin_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
origin_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 캠에 따라 FPS가 0으로 찍히는 경우가 있으므로 안정적인 30FPS로 강제 지정
if fps == 0 or fps > 60:
    fps = 30.0

# 2. [파일 이름 설정] 오직 시간 숫자로만 파일명 구성 (년월일_시분초.mp4)
# 예: 20260521_160512.mp4 형식으로 저장됩니다
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{current_time}.mp4"

# [녹화 설정]
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 저장을 위한 코덱
out = cv2.VideoWriter(output_path, fourcc, fps, (origin_w, origin_h))

cv2.namedWindow('Webcam Balanced Blur & Save', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Balanced Blur & Save', 1280, 720)

frame_count = 0
prev_mask = None

print(f"실시간 모자이크 및 녹화를 시작합니다 파일명: {output_path}")
print("종료하려면 화면을 클릭하고 ESC를 누르세요")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라로부터 프레임을 가져올 수 없습니다")
        break

    frame_count += 1

    # [최적화] AI 연산은 가볍게 20%(0.2) 화질로 다운스케일해서 초고속 처리 (캠 딜레이 방지)
    scale_factor = 0.2
    frame_low = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    h, w, _ = frame_low.shape

    # 2프레임마다 연산 부하 분할
    if frame_count % 2 == 1 or prev_mask is None:
        results = model(frame_low, stream=True, classes=[0], conf=0.35, imgsz=192, verbose=False)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        has_mask = False

        for r in results:
            if r.masks is not None:
                has_mask = True
                for mask in r.masks.data:
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask = cv2.bitwise_or(combined_mask, (mask_resized > 0.4).astype(np.uint8) * 255)
        
        prev_mask = combined_mask if has_mask else None
    else:
        combined_mask = prev_mask
        has_mask = prev_mask is not None

    # 원본 해상도로 저장하기 위해 원본 프레임 복사
    final_frame = frame.copy()

    if has_mask:
        # [외곽 범위] 사람 외각선보다 살짝 크게 가리는 마스크 크기 지정
        dilation_kernel = np.ones((13, 5), np.uint8)
        expanded_mask_low = cv2.dilate(combined_mask, dilation_kernel, iterations=1)
        
        # 다운스케일된 마스크를 원본 캠 크기로 확대
        expanded_mask_large = cv2.resize(expanded_mask_low, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        # [블러 강도] 이목구비 식별을 막으면서 답답함을 해결한 황금 밸런스 값 적용 (71)
        blurred_frame = cv2.GaussianBlur(final_frame, (71, 71), 0)
        
        idx = (expanded_mask_large == 255)
        final_frame[idx] = blurred_frame[idx]

    # 3. [실시간 파일 저장] 모자이크된 캠 화면을 동영상 파일에 실시간으로 기록
    out.write(final_frame)

    # 4. [화면 출력] 모니터로 실시간 확인
    frame_output = cv2.resize(final_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Webcam Balanced Blur & Save', frame_output)

    # ESC 키를 누르면 안전하게 녹화를 마감하고 종료
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# 메모리 해제 및 파일 마감 (반드시 거쳐야 비디오 파일이 온전하게 저장됩니다)
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"녹화가 완료되었습니다 파일이 안전하게 저장되었습니다: {output_path}")