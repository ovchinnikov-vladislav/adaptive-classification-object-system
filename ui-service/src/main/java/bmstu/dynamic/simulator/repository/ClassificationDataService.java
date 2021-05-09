package bmstu.dynamic.simulator.repository;

import bmstu.dynamic.simulator.model.ClassificationData;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;

@Repository
public class ClassificationDataService {

    public Flux<ClassificationData> findByVideoId(String videoId) {
        //TODO: поиск в базе video_id
        return null;
    }

}