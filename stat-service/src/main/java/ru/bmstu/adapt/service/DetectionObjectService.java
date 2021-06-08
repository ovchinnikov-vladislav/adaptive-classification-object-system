package ru.bmstu.adapt.service;

import ru.bmstu.adapt.domain.DetectionObject;
import ru.bmstu.adapt.dto.DetectionObjectRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Slf4j
@RequiredArgsConstructor
@Service
public class DetectionObjectService {

    private final MongoTemplate mongoTemplate;

    public void save(String userId, String detectionProcessId, DetectionObjectRequest request) {
        DetectionObject detectionObject = DetectionObject.builder()
                .userId(userId)
                .detectionProcessId(detectionProcessId)
                .box(request.getBox())
                .clazz(request.getClazz())
                .score(request.getScore())
                .image(request.getImage())
                .numFrame(request.getIteration())
                .numObject(request.getNumObject())
                .createdDate(LocalDateTime.now())
                .build();

        mongoTemplate.save(detectionObject);
    }

    public void save(DetectionObject detectionObject) {
        mongoTemplate.save(detectionObject);
    }

    public DetectionObject findById(String id) {
        return mongoTemplate.findById(id, DetectionObject.class);
    }

    public List<DetectionObject> findByUserIdAndDetectionProcessId(String userId, String detectionProcessId) {
        Query query = new Query();
        query.addCriteria(new Criteria().andOperator(
                Criteria.where("userId").is(userId),
                Criteria.where("detectionProcessId").is(detectionProcessId)
        ));
        query.with(Sort.by(Sort.Direction.ASC, "createdDate"));

        List<DetectionObject> objects = mongoTemplate.find(query, DetectionObject.class);

        Set<Integer> resultIds = new HashSet<>();
        List<DetectionObject> result = new ArrayList<>();
        for (DetectionObject elem : objects) {
            if (!resultIds.contains(elem.getNumObject())) {
                resultIds.add(elem.getNumObject());
                result.add(elem);
            }
        }

        return result;
    }

    public List<DetectionObject> findByUserIdAndDetectionProcessIdAndNumObject(String userId, String detectionProcessId, Integer numObject) {
        Query query = new Query();
        query.addCriteria(new Criteria().andOperator(
                Criteria.where("userId").is(userId),
                Criteria.where("detectionProcessId").is(detectionProcessId),
                Criteria.where("numObject").is(numObject)
        ));
        query.with(Sort.by(Sort.Direction.ASC, "createdDate"));

        return mongoTemplate.find(query, DetectionObject.class);
    }
}
