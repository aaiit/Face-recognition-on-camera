# import the opencv library
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)


while(True):
	
	# Capture the video frame
	# by frame
	ret, new_frame = vid.read()


	# font = cv2.FONT_HERSHEY_SIMPLEX

	# cv2.putText(new_frame, "Hamza", (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('frame', new_frame)


	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
