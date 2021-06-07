package bmstu.dynamic.simulator.client;

import feign.Headers;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import reactor.core.publisher.Flux;

import java.util.stream.Stream;

@Headers({ "Accept: multipart/x-mixed-replace;boundary=frame" })
public interface MachineLearningClient {

    @GetMapping("/video_feed/{video_id}")
    Flux<Stream<byte[]>> getYoutubeVideoContent(@PathVariable("video_id") String videoId);

}
