package bmstu.dynamic.simulator.controller.advice;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.security.web.reactive.result.view.CsrfRequestDataValueProcessor;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.LocalDate;

@ControllerAdvice
public class CommonControllerAdvice {

    @ModelAttribute
    public void csrfToken(ServerWebExchange exchange, Model model) {
        model.addAttribute("csrf_name", CsrfRequestDataValueProcessor.DEFAULT_CSRF_ATTR_NAME);
        Mono<CsrfToken> csrfToken = exchange.getAttributeOrDefault(CsrfToken.class.getName(), Mono.empty());
        csrfToken.doOnSuccess(token -> model.addAttribute("csrf_token", token));
    }

    @ModelAttribute
    public void copyrightData(Model model) {
        model.addAttribute("now_year", LocalDate.now().getYear());
        model.addAttribute("copyright", "Владислав Овчинников");
    }

}
