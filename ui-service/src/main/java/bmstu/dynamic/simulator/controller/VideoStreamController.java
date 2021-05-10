package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.model.DetectionObject;
import bmstu.dynamic.simulator.service.VideoStreamService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.*;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.thymeleaf.spring5.context.webflux.IReactiveDataDriverContextVariable;
import org.thymeleaf.spring5.context.webflux.ReactiveDataDriverContextVariable;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/video")
@RequiredArgsConstructor
@Slf4j
public class VideoStreamController {

    private final VideoStreamService videoStreamService;

    @GetMapping("/video_content")
    public String videoContent(Model model) {
        return "fragments/video :: video_content";
    }

    @GetMapping("/youtube/{video_id}/{uuid_session}")
    public ResponseEntity<Flux<DataBuffer>> youtubeVideoByParams(@PathVariable(name = "video_id") String videoId,
                                                          @PathVariable(name = "uuid_session") String uuidSession,
                                                          @AuthenticationPrincipal OAuth2User oauth2User) {
        String userId = oauth2User.getName();

        Flux<DataBuffer> dataBufferFlux = videoStreamService.getYoutubeVideoContent(videoId, userId, uuidSession);

        HttpHeaders headers = HttpHeaders.writableHttpHeaders(HttpHeaders.EMPTY);
        headers.set(HttpHeaders.CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame");

        return new ResponseEntity<>(dataBufferFlux, headers, HttpStatus.OK);
    }

    @GetMapping("/camera/{uuid_session}")
    public ResponseEntity<Flux<DataBuffer>> cameraVideoByParams(@PathVariable(name = "uuid_session") String uuidSession,
                                                          @RequestParam(name = "video_addr") String videoAddr,
                                                          @AuthenticationPrincipal OAuth2User oauth2User) {
        String userId = oauth2User.getName();

        Flux<DataBuffer> dataBufferFlux = videoStreamService.getCameraVideoContent(videoAddr, userId, uuidSession);

        HttpHeaders headers = HttpHeaders.writableHttpHeaders(HttpHeaders.EMPTY);
        headers.set(HttpHeaders.CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame");

        return new ResponseEntity<>(dataBufferFlux, headers, HttpStatus.OK);
    }

    @GetMapping("/detection_objects/{uuid_session}")
    public String detectionObjects(Model model,
                                   @AuthenticationPrincipal OAuth2User oauth2User,
                                   @PathVariable(name = "uuid_session") String uuidSession) {
        String userId = oauth2User.getName();

        Flux<Map> detectionObjectsFlux = videoStreamService.getDetectionObjects(userId, uuidSession);

        IReactiveDataDriverContextVariable reactiveDataDrivenMode = new ReactiveDataDriverContextVariable(detectionObjectsFlux);

        model.addAttribute("detection_objects", reactiveDataDrivenMode);

        return "fragments/video/video_detection_object :: detection_objects_fragment";
    }

    @GetMapping("/detection_objects/{uuid_session}/{num_object}")
    public String detectionObjectPhotoByNumObject(Model model,
                                                  @AuthenticationPrincipal OAuth2User oauth2User,
                                                  @PathVariable(name = "uuid_session") String uuidSession,
                                                  @PathVariable(name = "num_object") Integer numObject) {
        String userId = oauth2User.getName();

        Flux<Map> detectionObjectsFlux = videoStreamService.getDetectionObjectsByNumObject(userId, uuidSession, numObject);

        IReactiveDataDriverContextVariable reactiveDataDrivenMode = new ReactiveDataDriverContextVariable(detectionObjectsFlux);

        model.addAttribute("num_object", numObject);
        model.addAttribute("detection_objects", reactiveDataDrivenMode);

        return "pages/video/all_photo_detection_object";
    }

}