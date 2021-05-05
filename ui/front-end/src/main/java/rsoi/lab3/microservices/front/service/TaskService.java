package rsoi.lab3.microservices.front.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import rsoi.lab3.microservices.front.client.GatewayClient;
import rsoi.lab3.microservices.front.entity.Status;
import rsoi.lab3.microservices.front.entity.task.Task;
import rsoi.lab3.microservices.front.model.TaskPage;

import java.util.UUID;

@Service
public class TaskService {

    private static final Logger logger = LoggerFactory.getLogger(TaskService.class);

    @Autowired
    private GatewayClient client;

    public Task findById(UUID id) {
        logger.info("findById() method called.");
        return client.findTaskById(id).orElse(null);
    }

    public Task findByUserIdAndTaskId(UUID idTask, String token) {
        logger.info("findByUserIdAndTaskId() method called.");
        return client.findTaskByUserIdAndTaskId(idTask, "Bearer " + token).orElse(null);
    }

    public TaskPage findAll(Integer page, Integer size) {
        logger.info("findAll() method called.");
        return client.findTaskAll(page, size);
    }

    public TaskPage findByUserId(UUID id, Integer page, Integer size, String token) {
        logger.info("findByUserId() method called.");
        return client.findTaskByUserId(page, size, "Bearer " + token);
    }

    public Task create(Task task, String token) {
        logger.info("create() method called.");
        task.setStatus(Status.ACTIVE);
        if (task.getTest() != null)
            task.getTest().setStatus(Status.ACTIVE);
        return client.createTask(task, "Bearer " + token).orElse(null);
    }

    public void update(UUID idTask, Task task, String token) {
        logger.info("update() method called.");
        task.setStatus(Status.ACTIVE);
        if (task.getTest() != null)
            task.getTest().setStatus(Status.ACTIVE);
        client.updateTask(idTask, task, "Bearer " + token);
    }

    public void delete(UUID idTask, String token) {
        logger.info("delete() method called.");
        client.deleteTask(idTask, "Bearer " + token);
    }
}