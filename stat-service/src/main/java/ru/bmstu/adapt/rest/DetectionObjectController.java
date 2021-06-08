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
@RequestMapping("/detection-objects")
@RequiredArgsConstructor
public class DetectionObjectController {

    private final DetectionObjectService service;

    @PostMapping("/{user_id}/{detection_process_id}")
    public void save(@PathVariable("user_id") String userId,
                     @PathVariable("detection_process_id") String detectionProcessId,
                     @RequestBody DetectionObjectRequest request) {
        service.save(userId, detectionProcessId, request);
    }

    @GetMapping("/{user_id}/{detection_process_id}")
    public List<DetectionObject> findByUserIdAndDetectionProcessId(@PathVariable("user_id") String userId,
                                                                   @PathVariable("detection_process_id") String detectionProcessId) {
        return service.findByUserIdAndDetectionProcessId(userId, detectionProcessId);
    }

    @GetMapping("/{user_id}/{detection_process_id}/{num_object}")
    public List<DetectionObject> findByUserIdAndDetectionProcessId(@PathVariable("user_id") String userId,
                                                                   @PathVariable("detection_process_id") String detectionProcessId,
                                                                   @PathVariable("num_object") Integer numObject) {
        return service.findByUserIdAndDetectionProcessIdAndNumObject(userId, detectionProcessId, numObject);
    }

}
