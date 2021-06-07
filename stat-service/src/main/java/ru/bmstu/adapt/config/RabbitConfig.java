package bmstu.dynamic.simulator.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.val;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Declarables;
import org.springframework.amqp.core.FanoutExchange;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.rabbit.annotation.RabbitListenerConfigurer;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.rabbit.listener.RabbitListenerEndpointRegistrar;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConversionException;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.messaging.handler.annotation.support.DefaultMessageHandlerMethodFactory;

@Configuration
public class RabbitConfig implements RabbitListenerConfigurer {

    public final static String STAT_FANOUT_QUEUE_NAME = "stat.fanout.queue";
    public final static String STAT_EXCHANGE_NAME = "stat.fanout.exchange";
    private static final boolean NON_DURABLE = false;
    private static final boolean DO_NOT_AUTO_DELETE = false;

    @Bean
    public Declarables fanoutBindings() {
        val fanoutUsereventsQueue = new Queue(STAT_FANOUT_QUEUE_NAME, NON_DURABLE);
        val fanoutExchange = new FanoutExchange(STAT_EXCHANGE_NAME, NON_DURABLE,
                DO_NOT_AUTO_DELETE);
        return new Declarables(fanoutUsereventsQueue, fanoutExchange,
                BindingBuilder.bind(fanoutUsereventsQueue).to(fanoutExchange));
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ObjectMapper objectMapper, ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(producerJackson2MessageConverter(objectMapper));
        return template;
    }

    @Bean
    public Jackson2JsonMessageConverter producerJackson2MessageConverter(ObjectMapper objectMapper) {
        return new CustomMessageConverter(objectMapper);
    }

    @Bean
    public DefaultMessageHandlerMethodFactory rabbitMessageHandler() {
        DefaultMessageHandlerMethodFactory factory = new DefaultMessageHandlerMethodFactory();
        factory.setMessageConverter(new MappingJackson2MessageConverter());
        return factory;
    }

    @Override
    public void configureRabbitListeners(RabbitListenerEndpointRegistrar registrar) {
        registrar.setMessageHandlerMethodFactory(rabbitMessageHandler());
    }

    public static class CustomMessageConverter extends Jackson2JsonMessageConverter {

        public CustomMessageConverter(ObjectMapper jsonObjectMapper) {
            super(jsonObjectMapper);
        }

        @Override
        public Object fromMessage(Message message) throws MessageConversionException {
            message.getMessageProperties().setContentType(MessageProperties.CONTENT_TYPE_JSON);
            return super.fromMessage(message);
        }
    }
}
