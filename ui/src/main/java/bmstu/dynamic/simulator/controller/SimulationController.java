package bmstu.dynamic.simulator.controller;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import java.io.File;

public class SimulationController {

    public ImageView imageView;
    public TextArea logsDetection;

    @FXML
    public void initialize() {
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

}
