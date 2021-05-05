package rsoi.lab3.microservices.front.entity.task;

import rsoi.lab3.microservices.front.entity.BaseEntity;
import rsoi.lab3.microservices.front.entity.test.Test;

import javax.validation.constraints.DecimalMax;
import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

public class Task extends BaseEntity implements Serializable {

    private UUID idUser;
    private String image;
    private Byte withoutTest;

    @NotEmpty
    @Size(max=255)
    private String nameTask;

    @Size(max=1000)
    private String description;

    @NotEmpty
    @Size(max=2500)
    private String textTask;

    @Size(max=10000)
    private String templateCode;

    @DecimalMin(value = "1") @DecimalMax(value = "5")
    private Byte complexity;

    private Test test;

    public UUID getIdUser() {
        return idUser;
    }

    public void setIdUser(UUID idUser) {
        this.idUser = idUser;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public Byte getWithoutTest() {
        return withoutTest;
    }

    public void setWithoutTest(Byte withoutTest) {
        this.withoutTest = withoutTest;
    }

    public String getNameTask() {
        return nameTask;
    }

    public void setNameTask(String nameTask) {
        this.nameTask = nameTask;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getTextTask() {
        return textTask;
    }

    public void setTextTask(String textTask) {
        this.textTask = textTask;
    }

    public String getTemplateCode() {
        return templateCode;
    }

    public void setTemplateCode(String templateCode) {
        this.templateCode = templateCode;
    }

    public Byte getComplexity() {
        return complexity;
    }

    public void setComplexity(Byte complexity) {
        this.complexity = complexity;
    }

    public Test getTest() {
        return test;
    }

    public void setTest(Test test) {
        this.test = test;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Task task = (Task) o;
        return Objects.equals(super.getId(), task.getId()) &&
                Objects.equals(idUser, task.idUser) &&
                Objects.equals(image, task.image) &&
                Objects.equals(withoutTest, task.withoutTest) &&
                Objects.equals(super.getCreatedAt(), task.getCreatedAt()) &&
                Objects.equals(super.getUpdatedAt(), task.getUpdatedAt()) &&
                Objects.equals(nameTask, task.nameTask) &&
                Objects.equals(description, task.description) &&
                Objects.equals(textTask, task.textTask) &&
                Objects.equals(templateCode, task.templateCode) &&
                Objects.equals(complexity, task.complexity) &&
                Objects.equals(test, task.test) &&
                Objects.equals(super.getStatus(), task.getStatus());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.getId(), idUser, image, withoutTest, super.getCreatedAt(), super.getUpdatedAt(),
                nameTask, description, textTask, templateCode, complexity, test);
    }

    @Override
    public String toString() {
        return "Task{" +
                "idTask=" + super.getId() +
                ", idUser=" + idUser +
                ", image='" + image + '\'' +
                ", withoutTest=" + withoutTest +
                ", createdAt=" + super.getCreatedAt() +
                ", updatedAt=" + super.getUpdatedAt() +
                ", nameTask='" + nameTask + '\'' +
                ", description='" + description + '\'' +
                ", textTask='" + textTask + '\'' +
                ", templateCode='" + templateCode + '\'' +
                ", complexity=" + complexity +
                ", test=" + test +
                ", status=" + getStatus() +
                '}';
    }
}
