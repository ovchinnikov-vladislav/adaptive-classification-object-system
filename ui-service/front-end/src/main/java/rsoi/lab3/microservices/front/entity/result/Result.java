package rsoi.lab3.microservices.front.entity.result;

import rsoi.lab3.microservices.front.entity.Status;

import javax.validation.constraints.DecimalMax;
import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.Digits;
import javax.validation.constraints.NotNull;
import java.io.Serializable;
import java.util.Date;
import java.util.Objects;
import java.util.UUID;

public class Result implements Serializable {

    private UUID idTask;
    private UUID idUser;
    private Date createdAt;
    private Date updatedAt;
    private Status status;

    @DecimalMin(value = "0")
    private Integer countAttempt;

    @NotNull
    @Digits(integer = 3, fraction = 2)
    @DecimalMin(value = "0") @DecimalMax(value = "100")
    private Double mark;

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

    public Date getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Date createdAt) {
        this.createdAt = createdAt;
    }

    public Date getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Date updatedAt) {
        this.updatedAt = updatedAt;
    }

    public Integer getCountAttempt() {
        return countAttempt;
    }

    public void setCountAttempt(Integer countAttempt) {
        this.countAttempt = countAttempt;
    }

    public Double getMark() {
        return mark;
    }

    public void setMark(Double mark) {
        this.mark = mark;
    }

    public Status getStatus() {
        return status;
    }

    public void setStatus(Status status) {
        this.status = status;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Result result = (Result) o;
        return Objects.equals(idTask, result.idTask) &&
                Objects.equals(idUser, result.idUser) &&
                Objects.equals(createdAt, result.createdAt) &&
                Objects.equals(updatedAt, result.updatedAt) &&
                Objects.equals(countAttempt, result.countAttempt) &&
                Objects.equals(mark, result.mark) &&
                Objects.equals(status, result.status);
    }

    @Override
    public int hashCode() {
        return Objects.hash(idTask, idUser, createdAt, updatedAt, countAttempt, mark, status);
    }

    @Override
    public String toString() {
        return "Result{" +
                "idTask=" + idTask +
                ", idUser=" + idUser +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                ", countAttempt=" + countAttempt +
                ", mark=" + mark +
                ", status=" + status +
                '}';
    }
}
