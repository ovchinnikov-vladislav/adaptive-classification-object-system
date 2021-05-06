package bmstu.dynamic.simulator.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/")
@RequiredArgsConstructor
public class ExampleComponentsController {

    @GetMapping("/blank")
    public String blank(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/blank";
    }

    @GetMapping("/buttons")
    public String buttons(Model model,
                          @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                          @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/buttons";
    }

    @GetMapping("/cards")
    public String cards(Model model,
                        @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                        @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/cards";
    }

    @GetMapping("/charts")
    public String charts(Model model,
                         @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                         @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/charts";
    }

    @GetMapping("/tables")
    public String tables(Model model,
                         @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                         @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/tables";
    }

    @GetMapping("/utilities-animation")
    public String utilitiesAnimation(Model model,
                                     @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                     @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/utilities-animation";
    }

    @GetMapping("/utilities-border")
    public String utilitiesBorder(Model model,
                                  @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                  @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/utilities-border";
    }

    @GetMapping("/utilities-color")
    public String utilitiesColor(Model model,
                                 @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                 @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/utilities-color";
    }

    @GetMapping("/utilities-other")
    public String utilitiesOther(Model model,
                                 @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                 @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_name", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "examples/utilities-other";
    }

}
