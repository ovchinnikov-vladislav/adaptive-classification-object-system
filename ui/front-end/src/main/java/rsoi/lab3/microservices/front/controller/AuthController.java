package rsoi.lab3.microservices.front.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import rsoi.lab3.microservices.front.entity.Status;
import rsoi.lab3.microservices.front.entity.user.Role;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.model.AuthorizationCode;
import rsoi.lab3.microservices.front.model.OAuthClient;
import rsoi.lab3.microservices.front.service.AuthService;
import rsoi.lab3.microservices.front.service.jwt.JwtTokenProvider;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import java.io.IOException;
import java.util.*;

@Controller
public class AuthController {

    @Autowired
    private AuthService authService;
    @Autowired
    private JwtTokenProvider jwtTokenProvider;

    @PostMapping(value = "/oauth/token", produces = MediaType.APPLICATION_JSON_UTF8_VALUE)
    public String logIn(@RequestBody HashMap<String, String> requestDto, HttpServletResponse response,
                        Model model) {
        Map<String, String> token = jwtTokenProvider.generationToken(requestDto);
        if (token.get("access_token") != null) {
            Cookie ut = new Cookie("ut", token.get("access_token"));
            ut.setPath("/");
            ut.setMaxAge(86400);
            response.addCookie(ut);
            Cookie rt = new Cookie("rt", token.get("refresh_token"));
            rt.setPath("/");
            response.addCookie(rt);
            rt.setMaxAge(86400);
            User user = jwtTokenProvider.getUserByToken(token.get("access_token"));
            if (user != null) {
                model.addAttribute("user", user);
            }
        }
        return "redirect:/";
    }

    @GetMapping(value = "/sign_in")
    public ModelAndView signIn(Model model, HttpServletRequest request) {
        request.getSession().setAttribute("resultTest", null);
        model.addAttribute("user", new User());
        return new ModelAndView("sign_in");
    }

    @PostMapping(value = "/sign_in")
    public String signIn(@ModelAttribute @Valid User user) {
        List<Role> roles = new ArrayList<>();
        Role role = new Role();
        role.setName("ROLE_USER");
        roles.add(role);
        user.setRoles(roles);
        user.setStatus(Status.ACTIVE);
        User u = authService.create(user);
        return "redirect:/";
    }

    @GetMapping(value = "/exit")
    public String exit(Model model,
                       @CookieValue(value = "ut", required = false) Cookie ut,
                       @CookieValue(value = "rt", required = false) Cookie rt,
                       HttpServletRequest request, HttpServletResponse response) {
        request.getSession().setAttribute("resultTest", null);
        model.addAttribute("user", new User());
        request.getSession().invalidate();
        if (ut != null) {
            ut.setValue(null);
            ut.setPath("/");
            response.addCookie(ut);
        }
        if (rt != null) {
            rt.setValue(null);
            rt.setPath("/");
            response.addCookie(rt);
        }
        return "redirect:/";
    }

    @GetMapping(value = "/oauth", produces = MediaType.APPLICATION_JSON_UTF8_VALUE)
    public String authorizeApp(@CookieValue(value = "ut", required = false) Cookie cookie,
                               @RequestParam(value = "response_type") String responseType,
                               @RequestParam(value = "client_id") String clientId,
                               @RequestParam(value = "redirect_uri") String redirectUri, Model model) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        if (!responseType.equals("code"))
            return "404";
        model.addAttribute("response_type", responseType);
        OAuthClient client;
        User user;
        try {
            client = authService.findClientById(UUID.fromString(clientId), cookie.getValue());
            user = jwtTokenProvider.getUserByToken(cookie.getValue());
            if (client == null)
                return "404";
        } catch (Exception exc) {
            return "404";
        }
        model.addAttribute("client", client);
        model.addAttribute("redirect_uri", redirectUri);
        model.addAttribute("user", user);
        return "proof_rights";
    }

    @GetMapping(value = "/oauth/login")
    public String login(@CookieValue(name = "ut", required = false) Cookie cookie) {
        if (cookie == null)
            return "login";
        if (cookie.getValue().equals(""))
            return "login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "login";
        return "redirect:/";
    }

    @PostMapping(value = "/oauth", produces = MediaType.APPLICATION_JSON_UTF8_VALUE)
    public ModelAndView getCode(@CookieValue(name = "ut", required = false) Cookie cookie,
                        @RequestParam(value = "response_type") String responseType,
                        @RequestParam(value = "client_id") UUID clientId,
                        @RequestParam(value = "redirect_uri") String redirectUri,
                        HttpServletResponse response) throws IOException {
        if (responseType.equals("code")) {
            String token = jwtTokenProvider.resolveToken(cookie);
            if (token != null && jwtTokenProvider.validateAccessToken(token)) {
                AuthorizationCode code = authService.getCode(clientId, redirectUri, token);
                return new ModelAndView("redirect:" + redirectUri + "?code=" + code.getCode());
            }
        }
        return new ModelAndView("redirect:/");
    }
}
