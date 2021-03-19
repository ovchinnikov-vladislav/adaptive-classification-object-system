package bmstu.dynamic.simulator.controller;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.GridPane;

public class MainController {

    public MenuItem createMenuItem;
    public MenuItem openMenuItem;
    public MenuItem saveMenuItem;
    public MenuItem settingsMenuItem;
    public MenuItem exitMenuItem;
    public MenuItem machineLearningMenuItem;
    public MenuItem resultLeaningMenuItem;
    public MenuItem statisticMenuItem;
    public MenuItem aboutMenuItem;

    public GridPane machineLearningPane;
    public GridPane resultLearningPane;
    public GridPane statisticPane;

    public void initialize() {
        initMainMenu();
        initModelMenu();
    }

    private void initMainMenu() {
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

        settingsMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+Alt+S"));
        settingsMenuItem.setOnAction(event -> {
            // TODO: заглушка
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Settings windows");
            alert.setContentText("This is example");
            alert.showAndWait();
        });

        exitMenuItem.setAccelerator(KeyCombination.keyCombination("Alt+Q"));
        exitMenuItem.setOnAction(event -> System.exit(0));
    }

    private void initModelMenu() {
        machineLearningMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+Alt+M"));
        machineLearningMenuItem.setOnAction(event -> {
            machineLearningPane.setVisible(true);
            resultLearningPane.setVisible(false);
            statisticPane.setVisible(false);
        });
        resultLeaningMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+Alt+R"));
        resultLeaningMenuItem.setOnAction(event -> {
            resultLearningPane.setVisible(true);
            machineLearningPane.setVisible(false);
            statisticPane.setVisible(false);
        });
        statisticMenuItem.setAccelerator(KeyCombination.keyCombination("Ctrl+Alt+I"));
        statisticMenuItem.setOnAction(event -> {
            statisticPane.setVisible(true);
            machineLearningPane.setVisible(false);
            resultLearningPane.setVisible(false);
        });
    }

}
