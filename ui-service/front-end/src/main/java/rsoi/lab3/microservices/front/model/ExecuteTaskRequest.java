package rsoi.lab3.microservices.front.model;

import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.NotNull;
import java.util.Objects;
import java.util.UUID;

public class ExecuteTaskRequest {

    @NotNull
    private UUID idTask;
    @NotNull
    private UUID idUser;

    @NotNull
    private String sourceTask;

    public UUID getIdTask() {
        return idTask;
    }

    public void setIdTask(UUID idTask) {
        this.idTask = idTask;
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExecuteTaskRequest that = (ExecuteTaskRequest) o;
        return Objects.equals(idTask, that.idTask) &&
                Objects.equals(idUser, that.idUser) &&
                Objects.equals(sourceTask, that.sourceTask);
    }

    @Override
    public int hashCode() {
        return Objects.hash(idTask, idUser, sourceTask);
    }

    @Override
    public String toString() {
        return "ExecuteTaskRequest{" +
                "idTask=" + idTask +
                ", idUser=" + idUser +
                ", sourceTask=" + sourceTask +
                '}';
    }
}
