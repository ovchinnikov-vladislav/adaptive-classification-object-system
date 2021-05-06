package rsoi.lab3.microservices.front.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import rsoi.lab3.microservices.front.client.GatewayClient;
import rsoi.lab3.microservices.front.entity.result.Result;
import rsoi.lab3.microservices.front.model.ExecuteTaskRequest;
import rsoi.lab3.microservices.front.model.ResultTest;

import java.util.UUID;

@Service
public class TaskExecutorService {

    private static final Logger log = LoggerFactory.getLogger(TaskExecutorService.class);

    @Autowired
    private GatewayClient client;

    public ResultTest execute(ExecuteTaskRequest executeTaskRequest, String token) {
        log.info("execute() method called.");
        return client.executeTask(executeTaskRequest, "Bearer " + token).orElse(null);
    }

    public Result findResultByUserIdAndTaskId(UUID idUser, UUID idTask, String token) {
        try {
            log.info("findResultByUserIdAndTaskId() method called.");
            return client.findResultByUserIdAndTaskId(idTask, "Bearer " + token).orElse(null);
        } catch (Exception exc) {
            log.error("\t" + exc.getMessage());
        }
        return null;
    }
}
