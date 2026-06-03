from djitellopy import Tello
import cv2
import time
import pygame
import traceback
import os

# ==========================================================
# Tello + pygame + OpenCV
# 顔検知版
#
# 変更点
# - 黄色ポール認識を削除
# - Aキーでの黄色ポールスピン処理を削除
# - Haar Cascade による顔検知を追加
# - 横顔検知を追加
# - 遠距離でも拾いやすいように感度を少し強めに調整
# - 検出した顔に枠、中心点、顔数を表示
# - Cascadeの感度は変えず、連続検出チェックで誤検知を削減
# ==========================================================

pygame.init()
screen = pygame.display.set_mode((560, 400))
pygame.display.set_caption("Tello Control - Face Detection")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

tello = None
is_flying = False

SAVE_DIR = "capture_data"
os.makedirs(SAVE_DIR, exist_ok=True)

recording = False
video_writer = None
video_filename = None

j_pressed_start = None
j_photo_taken_during_hold = False
J_HOLD_SECONDS = 3.0

u_pressed_start = None
u_action_done_during_hold = False
U_HOLD_SECONDS = 3.0

# =========================
# 顔検知 調整場所
# =========================
# 正面顔 + 横顔を使う
FRONTAL_FACE_CASCADE_FILE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
PROFILE_FACE_CASCADE_FILE = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")

# 数値を下げるほど検知しやすいが、誤検知も増える
# 遠くの顔を拾いやすくするため、少し強めに設定
FACE_SCALE_FACTOR = 1.08
FACE_MIN_NEIGHBORS = 4
PROFILE_MIN_NEIGHBORS = 4
FACE_MIN_SIZE = (22, 22)

# 近い重複枠をまとめる判定。大きいほど同じ顔としてまとめやすい
FACE_DUPLICATE_IOU_THRESHOLD = 0.25

# =========================
# 誤検知削減設定
# =========================
# Cascadeの感度はそのままにして、検出結果の採用だけ慎重にする。
# 同じ位置付近で何回か連続して出た顔だけ表示する。
# 1にすると従来通り即表示、2以上にすると誤検知に強いが少しだけ表示が遅れる。
FACE_CONFIRM_HITS = 2

# 一瞬だけ検出が途切れても顔を消さない猶予。大きすぎると古い枠が残りやすい。
FACE_TRACK_MAX_MISSES = 2

# 前回の顔と同じ顔か判定する距離。大きいほど追跡しやすいが、別の顔を混同しやすい。
FACE_TRACK_DISTANCE_RATIO = 0.75

# =========================
# 色による誤検知削減設定
# =========================
# 顔検知の感度は変えず、検出された候補だけを軽くチェックする。
# 壁やポスターなどを顔として拾った時に、肌色や髪の毛っぽい暗い領域が少なければ除外する。
USE_SKIN_HAIR_FILTER = True

# 顔候補の中に必要な肌色の割合。下げるほど通りやすい。
# 遠くの顔や照明差を考えて、厳しすぎない値にしている。
MIN_SKIN_RATIO = 0.035

# 顔候補の上側に必要な髪・眉・目などの暗い領域の割合。
# 髪が明るい人や帽子でも落ちすぎないよう低め。
MIN_DARK_UPPER_RATIO = 0.015

# 肌色だけ、または暗い上側だけでもある程度あれば通すための緩い条件。
# マスク、逆光、金髪などで落ちすぎるのを防ぐ。
SKIN_STRONG_RATIO = 0.075
DARK_UPPER_STRONG_RATIO = 0.055

# 小さすぎる候補は色判定が不安定なので、従来の検知を優先して通す。
COLOR_FILTER_MIN_FACE_SIZE = 28

# 横顔検知だけ間引く。
# 1 = 正面顔と同じ頻度, 2 = 正面検知2回に1回, 3 = 正面検知3回に1回
# 正面顔の検知速度は落とさず、横顔だけ低頻度にして軽くする。
PROFILE_DETECT_EVERY_N_DETECTIONS = 3

# =========================
# ラグ軽減設定
# =========================
# 1 = 毎フレーム, 2 = 2フレームに1回, 3 = 3フレームに1回
DETECT_EVERY_N_FRAMES = 2

# 認識処理を軽くするため、縮小画像で顔検知する
DETECT_SCALE = 0.70

# バッテリー取得は通信が重いので1秒に1回だけ
BATTERY_CHECK_INTERVAL = 1.0

# pygame側の文字画面更新も毎フレームではなく間引く
UI_UPDATE_INTERVAL = 0.15


frontal_face_cascade = cv2.CascadeClassifier(FRONTAL_FACE_CASCADE_FILE)
profile_face_cascade = cv2.CascadeClassifier(PROFILE_FACE_CASCADE_FILE)

if frontal_face_cascade.empty():
    raise RuntimeError(f"Failed to load face cascade: {FRONTAL_FACE_CASCADE_FILE}")
if profile_face_cascade.empty():
    raise RuntimeError(f"Failed to load profile cascade: {PROFILE_FACE_CASCADE_FILE}")


def draw_text_lines(screen, font, lines):
    screen.fill((30, 30, 30))
    for i, text in enumerate(lines):
        img = font.render(text, True, (255, 255, 255))
        screen.blit(img, (20, 15 + i * 24))
    pygame.display.flip()


def save_photo(frame_bgr):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
    cv2.imwrite(filename, frame_bgr)
    print(f"[PHOTO] Saved: {filename}")


def start_recording(frame_bgr):
    global recording, video_writer, video_filename

    if frame_bgr is None:
        print("[REC] Frame is None. Cannot start recording.")
        return

    height, width = frame_bgr.shape[:2]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(SAVE_DIR, f"video_{timestamp}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width, height))

    if not video_writer.isOpened():
        print("[REC] Failed to open video writer.")
        video_writer = None
        return

    recording = True
    print(f"[REC] START: {video_filename}")


def stop_recording():
    global recording, video_writer, video_filename

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    if recording:
        print(f"[REC] STOP: {video_filename}")

    recording = False
    video_filename = None


def get_countdown_number(held_time, hold_seconds):
    remain = hold_seconds - held_time
    if remain > 2:
        return 3
    elif remain > 1:
        return 2
    elif remain > 0:
        return 1
    else:
        return 0


def convert_tello_frame_to_bgr(frame):
    """
    djitellopyのframe_read.frameはRGBで来ることが多い。
    OpenCVはBGR前提なので、ここで変換する。
    """
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def calc_iou(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = a["area"] + b["area"] - intersection
    return intersection / union if union > 0 else 0


def merge_duplicate_faces(faces):
    """
    正面・横顔・反転横顔で同じ顔が複数出た場合に、大きい枠だけ残す。
    """
    faces.sort(key=lambda f: f["area"], reverse=True)
    merged = []
    for face in faces:
        if all(calc_iou(face, kept) < FACE_DUPLICATE_IOU_THRESHOLD for kept in merged):
            merged.append(face)
    return merged


def calc_skin_hair_score(raw_frame, face):
    """
    壁などの誤検知を減らすため、顔候補の色だけを軽く確認する。
    Haar Cascadeの検知感度は変更しない。
    """
    frame_h, frame_w = raw_frame.shape[:2]
    x = max(0, face["x"])
    y = max(0, face["y"])
    w = max(1, face["w"])
    h = max(1, face["h"])
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    roi = raw_frame[y:y2, x:x2]
    if roi.size == 0:
        return 0.0, 0.0

    # 遠距離で小さい顔は色がつぶれやすいので、色判定を弱める。
    if min(w, h) < COLOR_FILTER_MIN_FACE_SIZE:
        return 1.0, 1.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # 肌色判定: HSVだけだと照明に弱いので、YCrCbも併用する。
    # 日本人の肌色だけに寄せすぎず、広めに取る。
    hsv_skin = (
        (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) &
        (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 180) &
        (hsv[:, :, 2] >= 45)
    )
    ycrcb_skin = (
        (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 180) &
        (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 135)
    )
    skin_mask = hsv_skin | ycrcb_skin
    skin_ratio = float(skin_mask.mean())

    # 髪・眉・目などは顔の上側に暗い領域として出やすい。
    # 壁の模様はここが弱いことが多い。
    upper = hsv[: max(1, hsv.shape[0] // 2), :, :]
    dark_mask = (upper[:, :, 2] <= 95) & (upper[:, :, 1] >= 20)
    dark_upper_ratio = float(dark_mask.mean())

    return skin_ratio, dark_upper_ratio


def passes_skin_hair_filter(raw_frame, face):
    """
    肌色 + 髪/目/眉っぽい暗部で顔候補を確認する。
    どちらか一方だけで落としすぎないよう、通常条件と強め条件を併用する。
    """
    if not USE_SKIN_HAIR_FILTER:
        return True

    skin_ratio, dark_upper_ratio = calc_skin_hair_score(raw_frame, face)
    face["skin_ratio"] = skin_ratio
    face["dark_upper_ratio"] = dark_upper_ratio

    normal_ok = skin_ratio >= MIN_SKIN_RATIO and dark_upper_ratio >= MIN_DARK_UPPER_RATIO
    skin_only_ok = skin_ratio >= SKIN_STRONG_RATIO
    dark_only_ok = dark_upper_ratio >= DARK_UPPER_STRONG_RATIO
    return normal_ok or skin_only_ok or dark_only_ok


def filter_faces_by_skin_hair(raw_frame, faces):
    """
    顔検知後の候補だけを絞る。検知パラメータは変えない。処理はROI内だけなので軽め。
    """
    return [face for face in faces if passes_skin_hair_filter(raw_frame, face)]


def is_same_face_for_tracking(face, track):
    """
    Cascadeの検知感度は変えず、前回と近い位置の顔かだけを判定する。
    IoUまたは中心距離のどちらかが近ければ同じ顔として扱う。
    """
    if calc_iou(face, track) >= 0.15:
        return True

    dx = face["cx"] - track["cx"]
    dy = face["cy"] - track["cy"]
    distance = (dx * dx + dy * dy) ** 0.5
    limit = max(face["w"], face["h"], track["w"], track["h"]) * FACE_TRACK_DISTANCE_RATIO
    return distance <= limit


def update_stable_faces(detected_faces, face_tracks):
    """
    1回だけ出た誤検知を表示しないための簡易トラッキング。
    顔検知そのものの感度・サイズ・頻度は変えない。
    """
    updated_tracks = []
    used_track_indexes = set()

    for face in detected_faces:
        best_index = None
        best_score = -1.0

        for i, track in enumerate(face_tracks):
            if i in used_track_indexes:
                continue
            if not is_same_face_for_tracking(face, track):
                continue

            score = calc_iou(face, track)
            if score > best_score:
                best_score = score
                best_index = i

        if best_index is not None:
            old_track = face_tracks[best_index]
            face["hits"] = old_track.get("hits", 1) + 1
            face["misses"] = 0
            used_track_indexes.add(best_index)
        else:
            face["hits"] = 1
            face["misses"] = 0

        updated_tracks.append(face)

    # 今回見つからなかった顔も少しだけ残す。
    for i, track in enumerate(face_tracks):
        if i in used_track_indexes:
            continue
        misses = track.get("misses", 0) + 1
        if misses <= FACE_TRACK_MAX_MISSES:
            kept = track.copy()
            kept["misses"] = misses
            updated_tracks.append(kept)

    stable_faces = [
        track for track in updated_tracks
        if track.get("hits", 1) >= FACE_CONFIRM_HITS
    ]

    stable_faces.sort(key=lambda f: f["area"], reverse=True)
    return stable_faces, updated_tracks


def append_detected_faces(faces, detected, scale_back, face_type, frame_w):
    for (x, y, w, h) in detected:
        x0 = int(x * scale_back)
        y0 = int(y * scale_back)
        w0 = int(w * scale_back)
        h0 = int(h * scale_back)

        # 右向き横顔は、左右反転画像で検知して元座標に戻す
        if face_type == "profile_right":
            x0 = frame_w - x0 - w0

        cx = x0 + w0 // 2
        cy = y0 + h0 // 2
        faces.append({
            "x": x0,
            "y": y0,
            "w": w0,
            "h": h0,
            "cx": cx,
            "cy": cy,
            "area": w0 * h0,
            "type": face_type,
        })


def detect_faces(display_frame, raw_frame, detection_count, cached_profile_faces, face_tracks):
    """
    Haar Cascadeで正面顔と横顔を検知する。
    検知画像サイズと正面顔の検知頻度は落とさず、横顔だけ低頻度にする。
    """
    frame_h, frame_w = raw_frame.shape[:2]

    if DETECT_SCALE != 1.0:
        detect_frame = cv2.resize(
            raw_frame,
            None,
            fx=DETECT_SCALE,
            fy=DETECT_SCALE,
            interpolation=cv2.INTER_AREA
        )
    else:
        detect_frame = raw_frame

    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 正面顔は今まで通り毎回検知する
    frontal_detected = frontal_face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    scale_back = 1.0 / DETECT_SCALE

    front_faces = []
    append_detected_faces(front_faces, frontal_detected, scale_back, "front", frame_w)

    profile_faces = cached_profile_faces

    # 横顔だけ低頻度で検知する。ここが一番重いので、ラグ軽減に効く。
    if detection_count % PROFILE_DETECT_EVERY_N_DETECTIONS == 0:
        profile_left_detected = profile_face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=PROFILE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )

        # 横顔Cascadeは左向き用。右向きも拾うため、左右反転した画像にもかける。
        flipped_gray = cv2.flip(gray, 1)
        profile_right_detected = profile_face_cascade.detectMultiScale(
            flipped_gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=PROFILE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )

        profile_faces = []
        append_detected_faces(profile_faces, profile_left_detected, scale_back, "profile_left", frame_w)
        append_detected_faces(profile_faces, profile_right_detected, scale_back, "profile_right", frame_w)

    faces = front_faces + profile_faces
    faces = merge_duplicate_faces(faces)

    # 肌色・髪/目/眉っぽい暗部がほとんど無い候補は、壁などの誤検知として除外する。
    # Cascadeの検知感度は変更していない。
    faces = filter_faces_by_skin_hair(raw_frame, faces)

    stable_faces, face_tracks = update_stable_faces(faces, face_tracks)

    draw_faces(display_frame, stable_faces, frame_w, frame_h)
    return stable_faces, profile_faces, face_tracks


def draw_faces(display_frame, faces, frame_w, frame_h):
    if faces:
        for i, face in enumerate(faces, start=1):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cx, cy = face["cx"], face["cy"]

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                f"FACE {i} {face.get('type', '')}: {w}x{h}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        main_face = faces[0]
        cv2.drawMarker(display_frame, (frame_w // 2, frame_h // 2), (255, 255, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.line(display_frame, (frame_w // 2, frame_h // 2), (main_face["cx"], main_face["cy"]), (255, 255, 255), 2)
        cv2.putText(
            display_frame,
            f"FACE OK count:{len(faces)} target:({main_face['cx']},{main_face['cy']})",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            display_frame,
            "FACE NG",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


def draw_cached_faces(display_frame, faces):
    if faces:
        for i, face in enumerate(faces, start=1):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"FACE {i} {face.get('type', '')}", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "FACE NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


try:
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
    time.sleep(2)

    UP_DOWN_SPEED = 60
    YAW_SPEED = 70
    MOVE_SPEED = 60

    running = True
    face_count = 0

    frame_count = 0
    battery = 0
    last_battery_check = 0.0
    last_ui_update = 0.0
    cached_faces = []
    cached_profile_faces = []
    face_tracks = []
    detection_count = 0

    while running:
        frame = frame_read.frame
        frame_bgr = convert_tello_frame_to_bgr(frame)

        now = time.time()
        if now - last_battery_check >= BATTERY_CHECK_INTERVAL:
            battery = tello.get_battery()
            last_battery_check = now

        keys = pygame.key.get_pressed()

        display_frame = None
        raw_frame = None
        faces = []

        if frame_bgr is not None:
            raw_frame = frame_bgr.copy()
            display_frame = frame_bgr.copy()

            frame_count += 1
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                detection_count += 1
                cached_faces, cached_profile_faces, face_tracks = detect_faces(
                    display_frame,
                    raw_frame,
                    detection_count,
                    cached_profile_faces,
                    face_tracks
                )
            else:
                # 認識を飛ばしたフレームは、前回結果だけ軽く描画する
                draw_cached_faces(display_frame, cached_faces)

            faces = cached_faces
            face_count = len(faces)

            cv2.putText(
                display_frame,
                f"Battery: {battery}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if recording:
                cv2.putText(
                    display_frame,
                    "REC",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            cv2.putText(
                display_frame,
                f"Face Count: {face_count}",
                (10, 295),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # j を3秒長押しで写真撮影
        if keys[pygame.K_j]:
            if j_pressed_start is None:
                j_pressed_start = time.time()
                j_photo_taken_during_hold = False
            else:
                held_time = time.time() - j_pressed_start
                if held_time >= J_HOLD_SECONDS and not j_photo_taken_during_hold:
                    if frame_bgr is not None:
                        save_photo(frame_bgr)
                        j_photo_taken_during_hold = True
        else:
            j_pressed_start = None
            j_photo_taken_during_hold = False

        # u を3秒長押しで録画開始/停止
        if keys[pygame.K_u]:
            if u_pressed_start is None:
                u_pressed_start = time.time()
                u_action_done_during_hold = False
            else:
                held_time = time.time() - u_pressed_start
                if held_time >= U_HOLD_SECONDS and not u_action_done_during_hold:
                    if not recording:
                        if frame_bgr is not None:
                            start_recording(frame_bgr)
                        else:
                            print("[REC] Cannot start recording because frame is None.")
                    else:
                        stop_recording()
                    u_action_done_during_hold = True
        else:
            u_pressed_start = None
            u_action_done_during_hold = False

        if display_frame is not None:
            if j_pressed_start is not None and not j_photo_taken_during_hold:
                held_time = time.time() - j_pressed_start
                remain = max(0.0, J_HOLD_SECONDS - held_time)
                cv2.putText(
                    display_frame,
                    f"Hold J for Photo: {remain:.1f}s",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            if u_pressed_start is not None and not u_action_done_during_hold:
                held_time = time.time() - u_pressed_start
                countdown = get_countdown_number(held_time, U_HOLD_SECONDS)
                msg = "Hold U to START recording" if not recording else "Hold U to STOP recording"
                cv2.putText(
                    display_frame,
                    msg,
                    (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                if countdown > 0:
                    cv2.putText(
                        display_frame,
                        str(countdown),
                        (280, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 0, 255),
                        8
                    )

            cv2.imshow("Tello Camera - Face Detection", display_frame)

        cv2.waitKey(1)

        if recording and video_writer is not None and frame_bgr is not None:
            video_writer.write(frame_bgr)

        lines = [
            "Space : Takeoff / Land",
            "Esc   : Exit",
            "Up/Down    : Up / Down",
            "Left/Right : Rotate",
            "e / c      : Forward / Back",
            "s / f      : Left / Right",
            "w          : Left Up Diagonal",
            "x          : Left Down Diagonal",
            "r          : Right Up Diagonal",
            "v          : Right Down Diagonal",
            "j (hold 3s): Take Photo",
            "u (hold 3s): Start / Stop Recording",
            "Target     : Face + Skin/Hair Filter",
            f"Face Count : {face_count}",
            f"Battery    : {battery}%",
            f"Recording  : {'ON' if recording else 'OFF'}",
        ]
        if time.time() - last_ui_update >= UI_UPDATE_INTERVAL:
            draw_text_lines(screen, font, lines)
            last_ui_update = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    if not is_flying:
                        tello.takeoff()
                        is_flying = True
                        time.sleep(0.8)
                    else:
                        tello.send_rc_control(0, 0, 0, 0)
                        tello.land()
                        is_flying = False
                        time.sleep(0.8)

        # 飛行制御
        lr = 0
        fb = 0
        ud = 0
        yaw = 0

        if is_flying:
            if keys[pygame.K_UP]:
                ud = UP_DOWN_SPEED
            elif keys[pygame.K_DOWN]:
                ud = -UP_DOWN_SPEED

            if keys[pygame.K_LEFT]:
                yaw = -YAW_SPEED
            elif keys[pygame.K_RIGHT]:
                yaw = YAW_SPEED

            if keys[pygame.K_e]:
                fb = MOVE_SPEED
            elif keys[pygame.K_c]:
                fb = -MOVE_SPEED

            if keys[pygame.K_s]:
                lr = -MOVE_SPEED
            elif keys[pygame.K_f]:
                lr = MOVE_SPEED

            if keys[pygame.K_w]:
                lr = -MOVE_SPEED
                fb = MOVE_SPEED
            elif keys[pygame.K_x]:
                lr = -MOVE_SPEED
                fb = -MOVE_SPEED
            elif keys[pygame.K_r]:
                lr = MOVE_SPEED
                fb = MOVE_SPEED
            elif keys[pygame.K_v]:
                lr = MOVE_SPEED
                fb = -MOVE_SPEED

            tello.send_rc_control(lr, fb, ud, yaw)
        else:
            tello.send_rc_control(0, 0, 0, 0)

        clock.tick(30)

except Exception:
    traceback.print_exc()

finally:
    if video_writer is not None:
        video_writer.release()

    if tello:
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        try:
            if is_flying:
                tello.land()
        except Exception:
            pass
        try:
            tello.streamoff()
        except Exception:
            pass

    cv2.destroyAllWindows()
    pygame.quit()
