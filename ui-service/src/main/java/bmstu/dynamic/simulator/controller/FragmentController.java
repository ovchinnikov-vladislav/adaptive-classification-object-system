package bmstu.dynamic.simulator.controller;

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

@Controller
@RequestMapping("/fragment")
@RequiredArgsConstructor
public class FragmentController {

    @Value("${services.caps-visual.url}")
    private String capsVisualUrl;

    @GetMapping("/sidebar")
    public String sidebar() {
        return "fragments/sidebar";
    }

    @GetMapping("/logout")
    public String logout() {
        return "fragments/logout";
    }

    @GetMapping("/footer")
    public String footer() {
        return "fragments/footer";
    }

    @GetMapping("/topbar")
    public String topbar() {
        return "fragments/topbar";
    }

    @GetMapping("/index_content")
    public String indexContent() {
        return "fragments/index :: index_content";
    }

    @GetMapping("/profile_content")
    public String profileContent(Model model,
                                 @RegisteredOAuth2AuthorizedClient OAuth2AuthorizedClient authorizedClient,
                                 @AuthenticationPrincipal OAuth2User oauth2User) {
        model.addAttribute("user_id", oauth2User.getName());
        model.addAttribute("user_attributes", oauth2User.getAttributes());
        model.addAttribute("client_name", authorizedClient.getClientRegistration().getClientName());
        return "fragments/profile :: profile_content";
    }

    @GetMapping("/settings_content")
    public String settingsContent() {
        return "fragments/settings :: settings_content";
    }

    @GetMapping("/video_content")
    public String videoContent() {
        return "fragments/video :: video_content";
    }

    @GetMapping("/activity_content")
    public String lkErrorContent() {
        return "fragments/activity :: activity_content";
    }

    @GetMapping("/examples/buttons_content")
    public String buttonsContent() {
        return "fragments/examples/buttons :: buttons_content";
    }

    @GetMapping("/examples/charts_content")
    public String chartsContent() {
        return "fragments/examples/charts :: charts_content";
    }

    @GetMapping("/examples/tables_content")
    public String tablesContent() {
        return "fragments/examples/tables :: tables_content";
    }

    @GetMapping("/examples/animation_content")
    public String animationContent() {
        return "fragments/examples/utilities-animation :: animation_content";
    }

    @GetMapping("/examples/border_content")
    public String borderContent() {
        return "fragments/examples/utilities-border :: border_content";
    }

    @GetMapping("/examples/color_content")
    public String colorContent() {
        return "fragments/examples/utilities-color :: color_content";
    }

    @GetMapping("/examples/other_content")
    public String otherContent() {
        return "fragments/examples/utilities-other :: other_content";
    }

}
