package bmstu.dynamic.simulator.service;

import org.springframework.stereotype.Service;

@Service
public class AuthService {

//    public AccessTokenResponse register(RegistrationRequest registrationRequest) {
//        verifyCaptcha(registrationRequest.getCaptcha());
//
//        CredentialRepresentation credential = new CredentialRepresentation();
//        credential.setType(PASSWORD);
//        credential.setValue(registrationRequest.getPassword());
//        credential.setTemporary(false);
//
//        var email = registrationRequest.getEmail();
//
//        var usersApi = keycloakAdminClient.realm(properties.getRealm()).users();
//
//        try {
//            UserRepresentation user = getUserRepresentation(registrationRequest, credential);
//
//            Response result = usersApi.create(user);
//
//            if (result.getStatus() == HttpStatus.CONFLICT.value()) {
//                throw new UserAlreadyExistException("User with email " + email + " already exist");
//            }
//
//            if (result.getStatus() != HttpStatus.CREATED.value()) {
//                throw new RegistrationException("User not registered " + result.getStatus());
//            }
//
//            user.setId(CreatedResponseUtil.getCreatedId(result));
//
//            if (sendVerificationLink != null && sendVerificationLink) {
//                usersApi.get(user.getId()).sendVerifyEmail();
//            }
//
//            var refererCode = getReferrerCode();
//            if (sendReferralRegistrationEvent != null && sendReferralRegistrationEvent && isNotEmpty(refererCode)) {
//                eventPublisher.publishReferralRegistered(user, refererCode);
//            } else {
//                eventPublisher.publishUserRegistered(user);
//            }
//
//            if (Boolean.TRUE.equals(returnToken)) {
//                var accessTokenResponse = tokenApi.obtainToken(
//                        email, registrationRequest.getPassword(),
//                        PASSWORD, properties.getResource(),
//                        recaptchaSecret, (String) properties.getCredentials().get(SECRET)
//                );
//
//                return objectMapper.readValue(accessTokenResponse, AccessTokenResponse.class);
//            }
//
//            return new AccessTokenResponse();
//        } catch (UserAlreadyExistException e) {
//            // rethrow
//            throw e;
//        } catch (Exception e) {
//            log.error("registration exception ", e);
//            throw new RegistrationException(e);
//        }
//    }

}
