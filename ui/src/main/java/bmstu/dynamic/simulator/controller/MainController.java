package bmstu.dynamic.simulator.controller;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.MenuItem;
import javafx.scene.input.KeyCombination;

public class MainController {

    @FXML
    private MenuItem createMenuItem;

    @FXML
    private MenuItem openMenuItem;

    @FXML
    private MenuItem saveMenuItem;

    @FXML
    private MenuItem exitMenuItem;

    @FXML
    public void initialize() {
        createMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+Alt+N"));
        createMenuItem.setOnAction(event -> {
            // TODO: заглушка
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Create windows");
            alert.setContentText("This is example");
            alert.showAndWait();
        });

        openMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+O"));
        openMenuItem.setOnAction(event -> {
            // TODO: заглушка
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Open windows");
            alert.setContentText("This is example");
            alert.showAndWait();
        });

        saveMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+S"));
        saveMenuItem.setOnAction(event -> {
            // TODO: заглушка
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Save windows");
            alert.setContentText("This is example");
            alert.showAndWait();
        });

        exitMenuItem.setAccelerator(KeyCombination.keyCombination("Alt+Q"));
        exitMenuItem.setOnAction(event -> System.exit(0));
    }

}
