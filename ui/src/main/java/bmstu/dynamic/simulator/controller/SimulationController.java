package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.utils.OpenCVUtils;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.concurrent.*;

public class SimulationController {

    public ImageView videoImageView;
    public TextArea logsTextArea;

//    private final VideoCapture capture = new VideoCapture();

    @FXML
    public void initialize() {
//        capture.open(0);
//        videoImageView.setPreserveRatio(true);
//        if (capture.isOpened()) {
//            Runnable frameGrabber = () -> {
//                Mat frame = grabFrame();
//                Image imageToShow = OpenCVUtils.mat2Image(frame);
//                OpenCVUtils.onFXThread(videoImageView.imageProperty(), imageToShow);
//            };
//            ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();
//            timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
//        }


//        Thread thread = new Thread(() -> {
//            while (true) {
//                // TODO: не вариант получать изображение так как рендер не справляется с частым обновлением
//                File file = new File("../shot.jpg");
//                Image image = new Image(file.toURI().toString());
//                imageView.setImage(image);
//            }
//        });
//        thread.start();
    }

//    private Mat grabFrame() {
//        // init everything
//        Mat frame = new Mat();
//
//        // check if the capture is open
//        if (this.capture.isOpened()) {
//            try {
//                // read the current frame
//                this.capture.read(frame);
//
//                // if the frame is not empty, process it
//                if (!frame.empty()) {
//                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
//                }
//
//            } catch (Exception e) {
//                // log the error
//                System.err.println("Exception during the image elaboration: " + e);
//            }
//        }
//
//        return frame;
//    }

}
