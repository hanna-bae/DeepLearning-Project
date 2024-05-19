import numpy as np 

def analyze_tree(data):
    sorted_idx = np.argsort(data[:, 0])
    sorted_data = data[sorted_idx]

    label = sorted_data[:, 0]
    X = np.array(sorted_data[:, 1], dtype=np.float64)
    Y = np.array(sorted_data[:, 2], dtype=np.float64)
    W = np.array(sorted_data[:, 3], dtype=np.float64)
    H = np.array(sorted_data[:, 4], dtype=np.float64)
    n_data = len(label)

    # fantasy(공상적), social_withdrawal(대인관계 회피), doubtful(편집증적 경향), self_confusion(자아혼란, 불안정), low_will_to_live(낮은 삶의 의지)
    score = np.zeros((5))
    # 결과 해석
    result = ['HTP 검사 중 나무 항목에서는 인생과 성장에 대한 상징이 투사된다고 알려져 있습니다. \n여기에는 무의식 수준의 자기 개념과 자기상, 적응 정도, 성취 및 포부 등이 반영됩니다.\n']

    # 0:나무전체, 1:기둥, 2:수관, 3:가지, 4:뿌리, 5:나뭇잎, 6:꽃
    # 7:열매, 8:그네, 9:새, 10:다람쥐, 11:구름, 12:달, 13: 별

    tree_size = W[0] * H[0]
    pilar_size = W[1] * H[1]
    trunk_size = W[2] * H[2]
    branch_size = W[3] * H[3]
    root_size = W[4] * H[4]
    leaf_size = W[5] * H[5]
    flower_size = W[6] * H[6]
    fruit_size = W[7] * H[7]
    swing_size = W[8] * H[8]
    bird_size = W[9] * H[9]
    squirrel_size = W[1] * H[10]
    cloud_size = W[11] * H[11]
    moon_size = W[12] * H[12]
    star_size = W[13] * H[13]

    trunk_coords = (X[2], Y[2], W[2], H[2])
    leaf_coords = (X[5], Y[5], W[5], H[5])
    fruit_coords = (X[7], Y[7], W[7], H[7])

    """ a 내부에 b의 중심이 있는지 확인 """
    def coord(a, b):
        ax, ay, aw, ah = a
        bx, by, _, _ = b
        return (ax - aw / 2 <= bx <= ax + aw / 2) and \
              (ay - ah / 2 <= by <= ay + ah / 2)


    # 기둥 = 줄기 (성격의 기본 요소, 피검자의 감정, 기본적 힘과 내적인 자아 강도, 기본적인 심리적 힘에 대한 지표 제공)
    if sorted_data[label == 1] is None:      # 기둥이 없으면
        score[4] += 1                  # 삶의 의지 상실
        print("기둥이 없는 경우, 삶의 의지가 낮은 상태로 해석됩니다.")
        result.append("기둥이 없는 경우, 삶의 의지가 낮은 상태로 해석됩니다.")

    if W[1] > 0.4:         # 현저하게 굵은 기둥
        score[4] -= 1                  # 삶의 의지 강함
        score[0] += 1                           # 공상적
        print("기둥이 현저하게 굵은 경우, 삶의 의지가 강하고 공상적인 경향이 있다고 해석됩니다.")
        result.append("기둥이 현저하게 굵은 경우, 삶의 의지가 강하고 공상적인 경향이 있다고 해석됩니다.")

    elif W[1] > 0.1:         # 굵은 기둥
        score[4] -= 1                  # 삶의 의지 강함
        print("기둥이 굵은 경우, 에너지가 높고 삶의 의지가 강하다고 해석됩니다.")
        result.append("기둥이 굵은 경우, 에너지가 높고 삶의 의지가 강하다고 해석됩니다.")

    elif W[1] < 0.05:      # 좁은 기둥
        score[4] += 1                  # 삶의 의지 부족
        print("기둥이 좁은 경우, 에너지가 낮고 삶의 의지가 낮다고 해석됩니다.")
        result.append("기둥이 좁은 경우, 에너지가 낮고 삶의 의지가 낮다고 해석됩니다.")


    # 가지 (주어진 환경으로부터 어떤 만족이나 원하는 것을 성취하려는 것을 표현, 사람 그림에서 팔과 무의식적인 유사성 지님, 수검자 자신이 지닌 능력을 나타냄)
    if sorted_data[label == 3] is None:
        score[1] += 1
        print("가지가 없는 경우, 타인과의 상호작용하며 즐거움을 나눈 경험이 거의 없다고 해석됩니다.")
        result.append("가지가 없는 경우, 타인과의 상호작용하며 즐거움을 나눈 경험이 거의 없다고 해석됩니다.")

    if branch_size > pilar_size * 1:     # 가지가 줄기에 비해 과도하게 클 때
        score[3] += 1                                       # 불안정감
        print("가지가 줄기에 비해 과도하게 큰 경우, 불안정감과 자아혼란을 많이 느끼는 상태로 해석됩니다.")
        result.append("가지가 줄기에 비해 과도하게 큰 경우, 불안정감과 자아혼란을 많이 느끼는 상태로 해석됩니다.")

    if branch_size > pilar_size * 0.3:   # 가지가 줄기에 비해 과도하게 작을 때
        score[2] += 1                               # 편집증적 경향
        score[3] += 1                         # 불안정감
        print("가지가 줄기에 비해 매우 작은 경우, 좌절감과 부적절감을 많이 느끼는 상태라 해석됩니다.")
        result.append("가지가 줄기에 비해 매우 작은 경우, 좌절감과 부적절감을 많이 느끼는 상태라 해석됩니다.")

    if H[3]/W[3] > 1:    # width / height. 과도하게 위를 향한 가지
        score[0] += 1               # 공상적
        score[2] += 1              # 편집증적 경향
        print("가지가 과도하게 위를 향한 경우, 현실보다는 공상이나 사물에 만족하면서 현실에서 만족을 얻기 두려운 상태라 해석됩니다.")
        result.append("가지가 과도하게 위를 향한 경우, 현실보다는 공상이나 사물에 만족하면서 현실에서 만족을 얻기 두려운 상태라 해석됩니다.")


    # 뿌리 (성격적 안정성, 안전에 대한 욕구, 현실과의 접촉 정도)
    if sorted_data[label == 4] is None:            # 뿌리가 없을 때
        score[3] += 1                  # 불안정감
        print("뿌리가 없는 경우, 불안정감과 부적절감을 가지고 있다고 해석됩니다.")
        result.append("뿌리가 없는 경우, 불안정감과 부적절감을 가지고 있다고 해석됩니다.")

    if Y[4] > 0.9:      # 뿌리가 가장자리에 위치
        score[3] += 1                  # 불안정감
        print("뿌리가 가장자리에 위치한 경우, 불안정하고 안정에 대한 욕구가 큰 상태라 해석됩니다.")
        result.append("뿌리가 가장자리에 위치한 경우, 불안정하고 안정에 대한 욕구가 큰 상태라 해석됩니다.")


    # 잎 (정신, 활력을 표현)
    if sorted_data[label == 5] is None:        # 잎이 없으면
        score[3] += 1       # 자아혼란
        print("잎을 그리지 않은 경우, 자아혼란을 느끼는 상태라 해석됩니다.")
        result.append("잎을 그리지 않은 경우, 자아혼란을 느끼는 상태라 해석됩니다.")

    if coord(trunk_coords, leaf_coords):  # 수관에 잎이 위치하면
        score[3] -= 1             # 안정적
        print("입이 수관에 위치한 경우, 안정에 대한 욕구가 강하고 쾌활한 성격을 가졌다고 해석됩니다.")
        result.append("입이 수관에 위치한 경우, 안정에 대한 욕구가 강하고 쾌활한 성격을 가졌다고 해석됩니다.")

    if Y[5] > 0.8:           # 잎이 바닥(아래)에 위치하면
        score[3] += 1                  # 불안정감
        score[1] += 1               # 대인관계 어렵
        print("떨어진 잎을 그린 경우, 사회의 요구에 순응하기 어려우며, 불안정한 상태로 해석됩니다.")
        result.append("떨어진 잎을 그린 경우, 사회의 요구에 순응하기 어려우며, 불안정한 상태로 해석됩니다.")

    if sum(label == 5) > 3:
        score[4] -= 1
        print("다량의 잎을 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")
        result.append("다량의 잎을 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")

    if leaf_size > branch_size * 0.5:    # 가지에 비해 큰 잎
        score[3] += 1                                  # 불안정감
        score[1] -= 1                               # 대인관계 잘함
        print("가지에 비에 잎이 큰 경우, 표면적으로는 잘 적응하고 있으나, 내면적으로 부적절감과 무력감을 안고 있음을 나타냅니다.")
        result.append("가지에 비에 잎이 큰 경우, 표면적으로는 잘 적응하고 있으나, 내면적으로 부적절감과 무력감을 안고 있음을 나타냅니다.")


    # 그 외
    if not coord(trunk_coords, fruit_coords):  # 수관에 열매가 위치하지 않으면
        score[2] += 1             # 편집증적 경향
        print("떨어진 열매를 그린 경우, 자신이 거부되고 있다는 감정을 느끼며 체념하고 있는 상태로 해석됩니다.")
        result.append("떨어진 열매를 그린 경우, 자신이 거부되고 있다는 감정을 느끼며 체념하고 있는 상태로 해석됩니다.")

    elif sum(label == 7) > 0:
        score[4] -= 1
        print("열매를 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")
        result.append("열매를 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")

    if sum(label == 6) > 0:
        score[4] -= 1
        print("꽃을 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")
        result.append("꽃을 그린 경우, 삶의 의지가 높은 상태로 해석됩니다.")

    if sum(label == 10) > 0:
        score[3] += 1
        print("다람쥐를 그린 경우, 연속적인 박탈 경험이 있어 불안정한 상태로 해석됩니다.")
        result.append("다람쥐를 그린 경우, 연속적인 박탈 경험이 있어 불안정한 상태로 해석됩니다.")

    # fantasy(공상적), social_withdrawal(대인과계 회피), doubtful(편집증적 경향), self_confusion(자아혼란, 불안정), low_will_to_live(낮은 삶의 의지)

    if score[0] == 0:
      result.append('공상적이지 않고 현실적이며 자기 만족을 잘 찾는 사람으로 보입니다. ')
    if score[1] == 0:
      result.append('대인관계에서 회피적이지 않고 적극적이며 사교적인 사람으로 보입니다. ')
    if score[2] == 0:
      result.append('편집증적 경향이 없으며 신뢰감을 가지고 있는 사람으로 보입니다. ')
    if score[3] == 0:
      result.append('자아혼란이나 불안정이 없고 자신을 명확히 이해하고 있는 사람으로 보입니다. ')
    if score[4] == 0:
      result.append('삶의 의지가 높고 활력 있으며 삶을 긍정적으로 바라보는 사람으로 보입니다. ')


    return score, result