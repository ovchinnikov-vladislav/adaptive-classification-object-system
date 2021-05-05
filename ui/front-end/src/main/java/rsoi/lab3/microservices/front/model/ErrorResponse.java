package rsoi.lab3.microservices.front.model;

import java.io.Serializable;
import java.util.Date;
import java.util.Objects;

public class ErrorResponse implements Serializable {
    private int status;
    private String error;
    private String message;
    private Date timestamp;
    private String path;

    public ErrorResponse() {
        timestamp = new Date();
    }

    public ErrorResponse(int status, String error) {
        this.status = status;
        this.error = error;
        this.timestamp = new Date();
    }

    public ErrorResponse(int status, String error, String message) {
        this.status = status;
        this.error = error;
        this.message = message;
        this.timestamp = new Date();
    }

    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ErrorResponse that = (ErrorResponse) o;
        return status == that.status &&
                Objects.equals(error, that.error) &&
                Objects.equals(message, that.message) &&
                Objects.equals(timestamp, that.timestamp) &&
                Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(status, error, message, timestamp, path);
    }

    @Override
    public String toString() {
        return "ErrorResponse{" +
                "status=" + status +
                ", error='" + error + '\'' +
                ", message='" + message + '\'' +
                ", timestamp=" + timestamp +
                ", path=" + path +
                '}';
    }
}
