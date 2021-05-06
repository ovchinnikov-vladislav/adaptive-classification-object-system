package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.service.ModelService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/")
@RequiredArgsConstructor
public class MainController {

    private final ModelService modelService;

    @GetMapping("/")
    public String index(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "index";
    }

    @GetMapping("/profile")
    public String profile(Model model,
                          @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                          @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "profile";
    }

    @GetMapping("/settings")
    public String settings(Model model,
                           @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                           @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "settings";
    }

    @GetMapping("/activity")
    public String activity(Model model,
                           @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                           @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "activity";
    }

    @GetMapping("/lk_error/{status}")
    public String lkError(@PathVariable Integer status, Model model,
                          @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                          @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        modelService.prepareErrorModelForTemplate(model, status);
        return "lk_error";
    }

    @GetMapping("/error/{status}")
    public String error(@PathVariable Integer status, Model model) {
        modelService.prepareErrorModelForTemplate(model, status);
        return "error";
    }
}
