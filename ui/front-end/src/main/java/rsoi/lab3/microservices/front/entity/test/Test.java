package rsoi.lab3.microservices.front.entity.test;

import rsoi.lab3.microservices.front.entity.BaseEntity;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

public class Test extends BaseEntity implements Serializable {

    private UUID idTask;
    private UUID idUser;

    @NotEmpty
    @Size(max=10000)
    private String sourceCode;

    @Size(max=1000)
    private String description;

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

    public String getSourceCode() {
        return sourceCode;
    }

    public void setSourceCode(String sourceCode) {
        this.sourceCode = sourceCode;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Test test = (Test) o;
        return Objects.equals(super.getId(), test.getId()) &&
                Objects.equals(idTask, test.idTask) &&
                Objects.equals(idUser, test.idUser) &&
                Objects.equals(super.getCreatedAt(), test.getCreatedAt()) &&
                Objects.equals(super.getUpdatedAt(), test.getUpdatedAt()) &&
                Objects.equals(sourceCode, test.sourceCode) &&
                Objects.equals(description, test.description) &&
                Objects.equals(super.getStatus(), test.getStatus());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.getId(), idTask, idUser, super.getCreatedAt(), super.getUpdatedAt(), sourceCode, description);
    }

    @Override
    public String toString() {
        return "Test{" +
                "idTest=" + super.getId() +
                ", idTask=" + idTask +
                ", idUser=" + idUser +
                ", createdAt=" + super.getCreatedAt() +
                ", updatedAt=" + super.getUpdatedAt() +
                ", sourceCode='" + sourceCode + '\'' +
                ", description='" + description + '\'' +
                ", status=" + super.getStatus() +
                '}';
    }
}
