package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.service.ModelService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.security.web.reactive.result.view.CsrfRequestDataValueProcessor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

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

    @GetMapping("/blank")
    public String blank(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "blank";
    }

    @GetMapping("/buttons")
    public String buttons(Model model,
                @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                @AuthenticationPrincipal OAuth2User oauth2User) {
            model.addAttribute("user_name", oauth2User.getName());
            model.addAttribute("user_attributes", oauth2User.getAttributes());
            model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "buttons";
    }

    @GetMapping("/cards")
    public String cards(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "cards";
    }

    @GetMapping("/charts")
    public String charts(Model model,
                         @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                         @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "charts";
    }

    @GetMapping("/tables")
    public String tables(Model model,
                         @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                         @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "tables";
    }

    @GetMapping("/utilities-animation")
    public String utilitiesAnimation(Model model,
                                     @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                     @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "utilities-animation";
    }

    @GetMapping("/utilities-border")
    public String utilitiesBorder(Model model,
                                  @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                  @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "utilities-border";
    }

    @GetMapping("/utilities-color")
    public String utilitiesColor(Model model,
                                 @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                 @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "utilities-color";
    }

    @GetMapping("/utilities-other")
    public String utilitiesOther(Model model,
                                 @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                 @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "utilities-other";
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
