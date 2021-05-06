package rsoi.lab3.microservices.front.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import rsoi.lab3.microservices.front.conf.FeignErrorDecoder;
import rsoi.lab3.microservices.front.entity.result.Result;
import rsoi.lab3.microservices.front.entity.task.Task;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.model.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@FeignClient(name="gateway-service", configuration = FeignErrorDecoder.class)
public interface GatewayClient {

    @GetMapping(value = "/api/v1/auth/clients/{id}")
    OAuthClient findClientById(@PathVariable UUID id, @RequestHeader("Authorization") String token);

    @PostMapping(value = "/api/v1/users")
    Optional<User> createUser(@RequestBody User user);

    @PostMapping(value = "/api/v1/auth/tasks/execute")
    Optional<ResultTest> executeTask(@RequestBody ExecuteTaskRequest request, @RequestHeader("Authorization") String token);

    @GetMapping(value = "/api/v1/auth/tasks/{idTask}/results")
    Optional<Result> findResultByUserIdAndTaskId(@PathVariable UUID idTask, @RequestHeader("Authorization") String token);

    @GetMapping(value = "/api/v1/tasks/{id}")
    Optional<Task> findTaskById(@PathVariable UUID id);

    @GetMapping(value = "/api/v1/auth/tasks/{idTask}")
    Optional<Task> findTaskByUserIdAndTaskId(@PathVariable UUID idTask, @RequestHeader("Authorization") String token);

    @GetMapping(value = "/api/v1/tasks")
    TaskPage findTaskAll(@RequestParam(value = "page") Integer page, @RequestParam(value = "size") Integer size);

    @GetMapping(value = "/api/v1/auth/tasks")
    TaskPage findTaskByUserId(@RequestParam(value = "page") Integer page,
                          @RequestParam(value = "size") Integer size, @RequestHeader("Authorization") String token);

    @PostMapping(value = "/api/v1/auth/tasks")
    Optional<Task> createTask(@RequestBody Task task, @RequestHeader("Authorization") String token);

    @PutMapping(value = "/api/v1/auth/tasks/{idTask}")
    void updateTask(@PathVariable UUID idTask, @RequestBody Task task, @RequestHeader("Authorization") String token);

    @DeleteMapping(value = "/api/v1/auth/tasks/{idTask}")
    void deleteTask(@PathVariable UUID idTask, @RequestHeader("Authorization") String token);

}
