import numpy as np

def read_txt(path):
  data = np.loadtxt(path, dtype=str)
  return data

def analyze_person(data):
    print(data)
    sorted_idx = np.argsort(data[:, 0])
    sorted_data = np.array(data[sorted_idx], dtype=np.float64)

    label = sorted_data[:, 0]
    X = sorted_data[:, 1]
    Y = sorted_data[:, 2]
    W = sorted_data[:, 3]
    H = sorted_data[:, 4]
    n_data = len(label)

    # aggresive, depressed, maladaptation, nervous
    score = np.zeros((4))
    # 결과 해석
    result = ['HTP 검사 중 사람 항목에서는 무의식적으로 자아가 투영되는 집과 나무와는 달리, 의식적 수준에서의 자아존중감과 대인관계에 대한 개인의 특성을 엿볼 수 있습니다. ']

    # 사람전체
    body_size = W[0] * H[0]
    if body_size > 0.8:
        score[0] += 1
        result.append('지면을 꽉 채우거나 밖으로 벗어날 정도로 크게 사람을 그린 경우, 충동적이며 공격적인 성향을 띨 가능성이 있습니다. ')
    elif body_size < 0.25:
        score[1] += 1
        result.append('지면의 1/4 이하로 작게 사람을 그렸다면 대인관계에서 무력감이나 열등감, 불안을 느끼고 있을 수 있습니다. ')
    if X[0] < 0.4:
        score[1] += 1
        result.append('좌측에 치우치게 사람을 그렸다면 내향적이고 소극적인 성격의 소유자일 수 있습니다. ')
    elif X[0] > 0.6:
        score[0] += 1
        result.append(('우측에 치우치게 사람을 그렸다면 외향적이고 이기적인 성격의 소유자일 수 있습니다. '))

    # 머리
    head_size = W[1] * H[1]
    if head_size/body_size > 0.3:
        score[0] += 1
        result.append('몸에 비해 지나치게 크게 머리를 그렸다면 자신을 과대평가하는 면이 있을 수 있습니다. ')
    elif head_size/body_size < 0.1:
        score[2] += 1
        result.append('몸에 비해 지나치게 작게 머리를 그렸다면 사회에 적응하는데 어려움을 겪거나 열등감을 갖고 있을 수 있습니다. ')

    # 눈
    face_size = W[2] * H[2]
    eye_ratio = np.zeros(4)
    for i in range(len(sorted_data[label == 3])):
        eye_ratio[i] = sorted_data[label == 3][i][3] * sorted_data[label == 3][i][4]
    eye_ratio /= face_size
    if sum(eye_ratio) > 0.03*2:
        score[3] += 1
        result.append('얼굴에 비해 큰 눈은 호기심이 많고 외향적인 성격임을 드러냅니다. 정도가 심할 경우, 주변을 경계하거나 예민하며 불안과 긴장감을 느끼고 있다는 징후일 수 있습니다. ')
    elif sum(eye_ratio) < 0.01*2:
        score[1] += 1
        result.append('얼굴에 비해 작은 눈은 내성적이고 관계 회피적인 성격임을 드러냅니다. ')

    # 코
    if sorted_data[label == 4] is None:
        score[3] += 1
        result.append('그림에서 코를 생략했다면 자신이 남에게 어떻게 보일지에 매우 예민하고 두려워하고 있을 수 있습니다. ')

    # 입
    if sorted_data[label == 5] is None:
        score[1] += 1
        result.append('입을 생략하여 그렸다면 타인과의 소통에 불안을 느끼고 있다는 징후일 수 있습니다. ')
    for i in sorted_data[label == 5]:
        mouth_size = i[3]*i[4]
        if mouth_size/face_size > 0.3:
            score[0] += 1
            result.append(('얼굴에 비해 입을 크게 그렸거나 강조하였다면 거친 언어를 사용하거나 욕심이 많다는 뜻일 수 있습니다. '))

    # 팔
    print(sorted_data[label == 9])
    torso_size = sorted_data[label == 9][0][3] * sorted_data[label == 9][0][4]
    if len(sorted_data[label == 10]) < 2:
        score[2] += 1
        result.append('팔을 그리지 않은 경우, 사람들과 관계를 맺고 싶어 하지만 내적인 갈등이 있거나 대인관계를 기피하거나 무력감을 느끼고 있다는 뜻일 수 있습니다. 또는 과도한 업무에 시달리고 있을 수 있습니다. ')
    arm_ratio = np.zeros(4)
    for i in range(len(sorted_data[label == 10])):
        arm_ratio[i] = sorted_data[label == 10][i][3] * sorted_data[label == 10][i][4]
    arm_ratio /= torso_size
    if sum(arm_ratio) > 0.6*2:
        score[0] += 1
        result.append('상체에 비하여 팔을 크게 그렸거나 강조하였다면 힘에 대한 욕구를 갖고 있는 사람일 수 있습니다. ')

# 손
    if len(sorted_data[label == 11]) < 2:
        score[2] += 1
        result.append('한 손만 그렸거나 손을 그리지 않았다면 다른 사람과 교류하고 싶은 소망은 있지만 스스로 할 수 없는 상태를 불안해하고 있는 것일 수 있습니다. ')

    # 발
    if len(sorted_data[label == 13]) < 2:
        score[3] += 1
        result.append('발을 그리지 않았다면 스스로 해야 한다는 것에 대한 불안감을 갖고 있다고 해석할 수 있습니다. ')

    # 주머니
    for i in sorted_data[label == 15]:
        pocket_size = i[3] * i[4]
        if pocket_size/torso_size > 0.3:
            score[1] += 1
            result.append('주머니를 강조하여 그렸다면 의존적이고 낮은 자존감을 갖고 있을 수 있습니다. ')

    if score[0] == 0:
      result.append('충동적이거나 공격적이지 않고 욕구와 분노를 원만하게 해결하는 사람으로 보입니다. ')
    if score[1] == 0:
      result.append('우울감이 적으며 대인관계에 있어 내성적이거나 회피적이지 않고 적극적인 사람으로 보입니다. ')
    if score[2] == 0:
      result.append('내적 갈등이나 억압을 잘 다스릴 수 있고 열등감이 적은 편입니다. ')
    if score[3] == 0:
      result.append('심리 상태가 불안하거나 예민하지 않고 평정심을 유지하는 편입니다. ')

    # print(score)
    # print(*result, sep='\n')
    return score, result

