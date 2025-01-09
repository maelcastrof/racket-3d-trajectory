import cv2
import numpy as np

# Cargar la imagen original de la mesa de tenis
image_path = '../frames/before_impact.png'
image = cv2.imread(image_path)

# Obtener el tamaño original de la imagen
height, width, _ = image.shape

# Definir los puntos de la imagen (estos son los puntos que ya tienes)
image_points = np.array([
    [733, 735],   # P0 (Bottom-left corner)
    [1333, 630],  # P1 (Bottom-right corner)
    [1876, 941],  # P2 (1/2 top-left corner)
    [1637, 649],  # P4 (1/2 top-right corner)
], dtype=np.float32)

# Dibujar las líneas entre P1 y P4
line_1_start = tuple(image_points[1].astype(int))  # P1
line_1_end = tuple(image_points[3].astype(int))    # P4
cv2.line(image, line_1_start, line_1_end, (0, 255, 0), 2)  # Línea 1 en verde

# Función de callback para dibujar una línea manualmente
drawing = False
pt1 = (0, 0)
pt2 = (0, 0)

# Función para manejar el clic y dibujar la línea
def draw_line(event, x, y, flags, param):
    global pt1, pt2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Si no se ha dibujado la primera línea, es el primer punto
        if not drawing:
            pt1 = (x, y)
            drawing = True
        else:
            # Si ya se ha dibujado el primer punto, es el segundo
            pt2 = (x, y)
            drawing = False
            # Dibujar la línea en la imagen
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Línea manual en azul

            # Mostrar la imagen con la línea dibujada
            cv2.imshow('Imagen con líneas', image)

# Crear una ventana de OpenCV con tamaño ajustable
cv2.namedWindow('Imagen con líneas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen con líneas', width, height)  # Ajustar el tamaño de la ventana al tamaño de la imagen

# Mostrar la imagen y esperar a que se dibuje la línea manualmente
cv2.imshow('Imagen con líneas', image)
cv2.setMouseCallback('Imagen con líneas', draw_line)

# Esperar hasta que el usuario presione una tecla
cv2.waitKey(0)
cv2.destroyAllWindows()

# Estimar la intersección de las dos líneas (P1-P4 y la línea manual)
# Línea 1: P1 a P4
m1 = (image_points[3][1] - image_points[1][1]) / (image_points[3][0] - image_points[1][0])
b1 = image_points[1][1] - m1 * image_points[1][0]

# Línea manual: pt1 a pt2
m2 = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
b2 = pt1[1] - m2 * pt1[0]

# Calcular la intersección de las dos líneas
x_intersect = (b2 - b1) / (m1 - m2)
y_intersect = m1 * x_intersect + b1

# Imprimir las coordenadas del punto estimado
print(f"Coordenadas del punto estimado: X: {x_intersect:.2f}, Y: {y_intersect:.2f}")
