module main {
    exports bmstu.dynamic.simulator;
    opens bmstu.dynamic.simulator to javafx.graphics;
    opens bmstu.dynamic.simulator.controller to javafx.fxml;

    requires javafx.fxml;
    requires javafx.controls;
}