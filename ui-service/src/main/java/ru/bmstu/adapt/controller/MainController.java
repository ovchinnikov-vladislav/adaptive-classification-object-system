package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.service.ModelService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

@Controller
@RequestMapping("/")
@RequiredArgsConstructor
@Slf4j
public class MainController {

    private final ModelService modelService;

    @GetMapping("/")
    public String index(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());

        return "layout";
    }

    @GetMapping("/profile")
    public String profile(Model model,
                          @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                          @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "pages/profile";
    }

    @GetMapping("/settings")
    public String settings(Model model,
                           @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                           @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());

        return "pages/settings";
    }

    @GetMapping("/activity")
    public String activity(Model model,
                           @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                           @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "pages/activity";
    }

}
