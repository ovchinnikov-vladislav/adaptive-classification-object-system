package ru.bmstu.adapt.rest;

import ru.bmstu.adapt.dto.DetectionObjectRequest;
import ru.bmstu.adapt.domain.DetectionObject;
import ru.bmstu.adapt.service.DetectionObjectService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/stat")
@RequiredArgsConstructor
public class StatClassificationController {

    private final DetectionObjectService service;

    @PostMapping("/users/{user_id}/process/{process_id}")
    public void save(@PathVariable("user_id") String userId,
                     @PathVariable("process_id") String processId,
                     @RequestBody DetectionObjectRequest request) {
        service.save(userId, processId, request);
    }

    @GetMapping("/users/{user_id}/process/{process_id}")
    public List<DetectionObject> findByUserIdAndDetectionProcessId(@PathVariable("user_id") String userId,
                                                                   @PathVariable("process_id") String detectionProcessId) {
        return service.findByUserIdAndDetectionProcessId(userId, detectionProcessId);
    }

    @GetMapping("/users/{user_id}/process/{process_id}/object/{num_object}")
    public List<DetectionObject> findByUserIdAndDetectionProcessId(@PathVariable("user_id") String userId,
                                                                   @PathVariable("process_id") String detectionProcessId,
                                                                   @PathVariable("num_object") Integer numObject) {
        return service.findByUserIdAndDetectionProcessIdAndNumObject(userId, detectionProcessId, numObject);
    }

    @GetMapping("/statistic-by-detection/users/{user-id}/")
    public List<DetectionObject> statisticByDetection() {
        return null;
    }

}
