package rsoi.lab3.microservices.front.model;

import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.NotNull;
import java.util.Objects;
import java.util.UUID;

public class ExecuteTask {

    @NotNull
    private UUID idTask;
    @NotNull
    private UUID idTest;
    @NotNull
    private UUID idUser;

    @NotNull
    private String sourceTask;

    @NotNull
    private String sourceTest;

    public UUID getIdTask() {
        return idTask;
    }

    public void setIdTask(UUID idTask) {
        this.idTask = idTask;
    }

    public UUID getIdTest() {
        return idTest;
    }

    public void setIdTest(UUID idTest) {
        this.idTest = idTest;
    }

    public UUID getIdUser() {
        return idUser;
    }

    public void setIdUser(UUID idUser) {
        this.idUser = idUser;
    }

    public String getSourceTask() {
        return sourceTask;
    }

    public void setSourceTask(String sourceTask) {
        this.sourceTask = sourceTask;
    }

    public String getSourceTest() {
        return sourceTest;
    }

    public void setSourceTest(String sourceTest) {
        this.sourceTest = sourceTest;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExecuteTask that = (ExecuteTask) o;
        return Objects.equals(idTask, that.idTask) &&
                Objects.equals(idTest, that.idTest) &&
                Objects.equals(idUser, that.idUser) &&
                Objects.equals(sourceTask, that.sourceTask) &&
                Objects.equals(sourceTest, that.sourceTest);
    }

    @Override
    public int hashCode() {
        return Objects.hash(idTask, idTest, idUser, sourceTask, sourceTest);
    }

    @Override
    public String toString() {
        return "ExecuteTaskRequest{" +
                "idTask=" + idTask +
                ", idTest=" + idTest +
                ", idUser=" + idUser +
                ", sourceTask=" + sourceTask +
                ", sourceTest=" + sourceTest +
                '}';
    }
}
