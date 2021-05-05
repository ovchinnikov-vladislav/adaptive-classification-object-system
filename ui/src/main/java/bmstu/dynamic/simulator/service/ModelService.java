package bmstu.dynamic.simulator.service;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.ui.Model;

import java.time.LocalDate;

@Service
public class ModelService {

    public void prepareStandardModelForTemplate(Model model) {
        model.addAttribute("now_year", LocalDate.now().getYear());
        model.addAttribute("copyright", "Владислав Овчинников");
    }

    public void prepareErrorModelForTemplate(Model model, Integer status) {
        model.addAttribute("status", "Неопределенная ошибка");

        if (HttpStatus.valueOf(status) == HttpStatus.NOT_FOUND) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Страница или ресурс не найдены");
        } else if (HttpStatus.valueOf(status) == HttpStatus.BAD_REQUEST) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Запрос недействителен");
        } else if (HttpStatus.valueOf(status).is5xxServerError()) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Фатальная ошибка (ошибка сервера)");
        }
        model.addAttribute("additional_text",  "Похоже вы нашли баг в матрице...");

        prepareStandardModelForTemplate(model);
    }
}
