import cv2
import numpy as np

def nothing(x):
    pass

# Cria uma janela chamada 'image'
cv2.namedWindow('image')

# Cria 6 barras deslizantes para os limites de HSV
cv2.createTrackbar('H_low', 'image', 0, 360, nothing)
cv2.createTrackbar('S_low', 'image', 0, 100, nothing)
cv2.createTrackbar('V_low', 'image', 0, 100, nothing)
cv2.createTrackbar('H_high', 'image', 360, 360, nothing)
cv2.createTrackbar('S_high', 'image', 100, 100, nothing)
cv2.createTrackbar('V_high', 'image', 100, 100, nothing)

# Carrega a imagem
image = cv2.imread('imgs/ex1.png')

# Converte a imagem para o espaço de cores HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # Obtém as posições atuais das barras deslizantes
    h_low = int(cv2.getTrackbarPos('H_low', 'image')/360*255)
    s_low = int(cv2.getTrackbarPos('S_low', 'image')/100*255)
    v_low = int(cv2.getTrackbarPos('V_low', 'image')/100*255)
    h_high = int(cv2.getTrackbarPos('H_high', 'image')/360*255)
    s_high = int(cv2.getTrackbarPos('S_high', 'image')/100*255)
    v_high = int(cv2.getTrackbarPos('V_high', 'image')/100*255)

    # Define os limites inferior e superior
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    # Cria uma máscara com os intervalos fornecidos
    mask = cv2.inRange(hsv_image, lower, upper)

    # Aplica a máscara à imagem original
    result = cv2.bitwise_and(image, image, mask=mask)

    # Mostra a imagem resultante
    cv2.imshow('image', result)

    # Sai do loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha todas as janelas
cv2.destroyAllWindows()
