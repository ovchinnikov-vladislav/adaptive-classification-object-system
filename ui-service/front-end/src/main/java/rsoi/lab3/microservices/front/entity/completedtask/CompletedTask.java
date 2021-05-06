package rsoi.lab3.microservices.front.entity.completedtask;

import rsoi.lab3.microservices.front.entity.BaseEntity;

import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

public class CompletedTask extends BaseEntity implements Serializable {

    private UUID idTask;
    private UUID idTest;
    private UUID idUser;
    private Byte wasSuccessful;

    @NotEmpty
    @Size(max=10000)
    private String sourceCode;

    @DecimalMin(value = "0")
    private Integer countSuccessfulTests;

    @DecimalMin(value = "0")
    private Integer countFailedTests;

    @DecimalMin(value = "0")
    private Integer countAllTests;

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

    public Byte getWasSuccessful() {
        return wasSuccessful;
    }

    public void setWasSuccessful(Byte wasSuccessful) {
        this.wasSuccessful = wasSuccessful;
    }

    public String getSourceCode() {
        return sourceCode;
    }

    public void setSourceCode(String sourceCode) {
        this.sourceCode = sourceCode;
    }

    public Integer getCountSuccessfulTests() {
        return countSuccessfulTests;
    }

    public void setCountSuccessfulTests(Integer countSuccessfulTests) {
        this.countSuccessfulTests = countSuccessfulTests;
    }

    public Integer getCountFailedTests() {
        return countFailedTests;
    }

    public void setCountFailedTests(Integer countFailedTests) {
        this.countFailedTests = countFailedTests;
    }

    public Integer getCountAllTests() {
        return countAllTests;
    }

    public void setCountAllTests(Integer countAllTests) {
        this.countAllTests = countAllTests;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CompletedTask that = (CompletedTask) o;
        return Objects.equals(super.getId(), that.getId()) &&
                Objects.equals(wasSuccessful, that.wasSuccessful) &&
                Objects.equals(super.getCreatedAt(), that.getCreatedAt()) &&
                Objects.equals(super.getUpdatedAt(), that.getUpdatedAt()) &&
                Objects.equals(sourceCode, that.sourceCode) &&
                Objects.equals(countSuccessfulTests, that.countSuccessfulTests) &&
                Objects.equals(countFailedTests, that.countFailedTests) &&
                Objects.equals(countAllTests, that.countAllTests) &&
                Objects.equals(idTask, that.idTask) &&
                Objects.equals(idTest, that.idTest) &&
                Objects.equals(idUser, that.idUser) &&
                Objects.equals(super.getStatus(), that.getStatus());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.getId(), wasSuccessful, super.getCreatedAt(), super.getUpdatedAt(),
                sourceCode, countSuccessfulTests, countFailedTests, countAllTests, idTask, idTest, idUser);
    }

    @Override
    public String toString() {
        return "CompletedTask{" +
                "idCompletedTask=" + super.getId() +
                ", wasSuccessful=" + wasSuccessful +
                ", createdAt=" + super.getCreatedAt() +
                ", updatedAt=" + super.getUpdatedAt() +
                ", sourceCode='" + sourceCode + '\'' +
                ", countSuccessfulTests=" + countSuccessfulTests +
                ", countFailedTests=" + countFailedTests +
                ", countAllTests=" + countAllTests +
                ", idTask=" + idTask +
                ", idTest=" + idTest +
                ", idUser=" + idUser +
                ", status=" + super.getStatus() +
                '}';
    }
}
