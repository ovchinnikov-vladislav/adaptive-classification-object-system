package ru.bmstu.adapt.controller;

import ru.bmstu.adapt.service.CapsVisualizationService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
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
@RequestMapping("/ml")
@RequiredArgsConstructor
public class MachineLearningController {

    private final CapsVisualizationService service;

    @Value("${services.caps-visual-url}")
    private String capsVisualUrl;

    @GetMapping("/capsnet")
    public Mono<String> capsnet(Model model,
                                @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                @AuthenticationPrincipal OAuth2User oauth2User) {
        return service.getVisualize().map((result) -> {
            model.addAttribute("user_id", oauth2User.getName());
            model.addAttribute("user_attributes", oauth2User.getAttributes());
            model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
            model.addAttribute("capsnet", capsVisualUrl);

            return "pages/capsnet";
        });
    }

    @GetMapping("/capsnet_content")
    public Mono<String> capsnetContent(Model model) {
        return service.getVisualize().map((result) -> {
            model.addAttribute("capsnet", capsVisualUrl);

            return "fragments/capsnet :: capsnet_content";
        });
    }

}
