package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.service.VideoStreamService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import reactor.core.publisher.Mono;

@Controller
@RequestMapping("/video")
@RequiredArgsConstructor
public class VideoStreamController {

    private final VideoStreamService videoStreamService;

    @GetMapping("/")
    public Mono<String> streamVideo(Model model,
                                    @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                    @AuthenticationPrincipal OAuth2User oauth2User) {
        return Mono.just(videoStreamService.prepareContent("test", "mp4", "Range")).map(result -> {
            model.addAttribute("user_id", oauth2User.getName());
            model.addAttribute("user_attributes", oauth2User.getAttributes());
            model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
            model.addAttribute("video", result);
            return "pages/video";
        });
    }

}