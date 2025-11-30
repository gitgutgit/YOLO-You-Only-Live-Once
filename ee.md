
한 이미지에 측정 되는 오브젝트 갯수 n개라고할떄

perfect score = n
score =0

label in labels:
    if label == correct:
        score += label.confidence 
    else:
        score -= label.confidence

total_score = score / perfect_score

예시 
game_20251122_BPrP-O6-_00720.jpg
실제로 메테오 7개 라바 1개  player 1개

메테오 같은경우에는 실제로 메테오 가 맞으니까
    score += label.confidence (0.99+0.98+0.97+0.96+0.95+0.94+0.93)

라바 같은경우데ㅗ 맞으니까
    score += label.confidence (0.99)

player 같은경우데 맞지만 좀 정확도 가낮으니까
    score += label.confidence (0.86)

star 같은경우네는 틀린거니까
    score -= label.confidence (0.44)


6.5 / 8=  0.8125    81.25 퍼센트 정확하다..