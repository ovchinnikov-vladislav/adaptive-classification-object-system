package bmstu.dynamic.simulator.rest;

import bmstu.dynamic.simulator.dto.ClassificationDataRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/classification-data")
@RequiredArgsConstructor
public class ClassificationDataController {

    @PostMapping("/{user_id}/{iteration}")
    public void save(@PathVariable("user_id") String userId, @PathVariable("iteration") String iteration,
                     @RequestBody ClassificationDataRequest request) {
        log.info("{}. {} -> {}", iteration, userId, request);
    }

}
