from verification.util.stance_detect import * 
from verification.util.sentiment import * 


class MultiView(torch.nn.Module):
    def __init__(self, bert, clip, concat_hidden_size, final_hidden_size, num_classes,max_image_num,model_type,model_type_enum,is_project_embed ,only_claim_setting,token_pooling_method):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiView, self).__init__()
        self.view1 = MultiModal2(bert = None, clip = clip, concat_hidden_size = 256, final_hidden_size = 64, 
                       num_classes = 3,max_image_num=max_image_num,model_type=model_type,model_type_enum=model_type_enum ,
                       is_project_embed=model_type_enum.is_project_embed,only_claim_setting=model_type_enum.only_claim_setting,
                       token_pooling_method=model_type_enum.token_pooling_method)
        self.view1.load_state_dict(torch.load('checkpoint/claim_verification/text_and_image_evidence/base.pt'),strict=False)
        self.view1.requires_grad_(False)

        self.max_image_num=self.view1.max_image_num
        self.model_type=self.view1.model_type
        self.model_type_enum=self.view1.model_type_enum
        self.is_project_embed=self.view1.is_project_embed

        self.view2 = Verifier(max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)
        self.view3 = Verifier(max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)

    
    def forward(self, claim_img_encod, confusing_claim_img_encod, random_claim_img_encod, device):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output_G = self.view1.feed_forward(claim_img_encod=claim_img_encod, device=device)
        has_image, claim_embed, txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask = self.view1.data_processing(claim_img_encod=confusing_claim_img_encod,device=device)
        output_C = self.view2(self.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        has_image, claim_embed, txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask = self.view1.data_processing(claim_img_encod=random_claim_img_encod,device=device)
        output_R = self.view3(self.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
     
        output_ = self.view2(self.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        return output_G, output_C - output_, output_R, output_C
    
    
    def inference(self, claim_img_encod,device):
        output_G, _, output_R, output_C = self.forward(claim_img_encod=claim_img_encod.copy(),confusing_claim_img_encod=claim_img_encod.copy(),random_claim_img_encod=claim_img_encod.copy(),device=device)
        # return output_G
        return self.view_fusion(output_G, output_C, output_R)
    
    def inference_ours(self, claim_img_encod,device):
        output_G, _, output_R, output_C = self.forward(claim_img_encod=claim_img_encod.copy(),confusing_claim_img_encod=claim_img_encod.copy(),random_claim_img_encod=claim_img_encod.copy(),device=device)
        # return output_G
        return self.view_fusion(output_G, output_C, output_R)
    
    def case_inference(self, claim_img_encod,device):
        output_G, _, output_R, output_C = self.forward(claim_img_encod=claim_img_encod.copy(),confusing_claim_img_encod=claim_img_encod.copy(),random_claim_img_encod=claim_img_encod.copy(),device=device)
        return output_G, output_G + output_C, output_G - output_R, self.view_fusion(output_G, output_C, output_R)
        # return self.view_fusion(output_G, output_C, output_R)
    
    # mean fusion
    def view_fusion(self,view1,view2,view3):
        # return 2*view1/3 - view3 / 3 + view2 / 3
        # return view1 - view3 + view2
        return view1 - view3/3 + view2/3
        return view1 + view2/2
        return view1 - view3/2 
    
    # max fusion
    # def view_fusion(self,view1,view2,view3):
    #     p1 = view1
    #     p2 = view1 + view2
    #     p3 = view1 - view3
    #     total_p = torch.cat([p1, p2, p3], dim=1)
    #     max_probability, max_index = torch.max(total_p, dim=1)
    #     if max_index < 3:
    #         return p1
    #     elif max_index < 6:
    #         return p2
    #     else:
    #         return p3
    
    # voting
    # def view_fusion(self,view1,view2,view3):
    #     p1 = view1
    #     p2 = view1 + view2
    #     p3 = view1 - view3
    #     max_probability1, max_index1 = torch.max(p1, dim=1)
    #     max_probability2, max_index2 = torch.max(p2, dim=1)
    #     max_probability3, max_index3 = torch.max(p3, dim=1)
    #     if max_index1 == max_index2 or max_index1 == max_index3:
    #         return p1
    #     elif max_index2 == max_index3:
    #         return p2
        
    #     if max_probability1 >= max_probability2 and max_probability1 >= max_probability3:
    #         return p1
    #     elif max_probability2 >= max_probability1 and max_probability2 >= max_probability3:
    #         return p2
    #     else:
    #         return p3
       
     
    
    # + view2
    # torch.tanh(view3)
        return view1 + torch.tanh(view2) - torch.tanh(view3)
        # return view1 + torch.tanh(view2) - torch.tanh(view3)

    



class MultiModal2(torch.nn.Module):
    
    def __init__(self, bert, clip, concat_hidden_size, final_hidden_size, num_classes,max_image_num,model_type,model_type_enum,is_project_embed ,only_claim_setting,token_pooling_method):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiModal2, self).__init__()
        
        self.clipmodel = clip
        self.max_image_num=max_image_num
        self.model_type=model_type
        self.model_type_enum=model_type_enum
        self.is_project_embed=is_project_embed
        self.verifier=Verifier( max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)
        # self.verifier_confusing=Verifier( max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)
        # self.verifier_random=Verifier( max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)
        

    # def forward(self, claim_img_encod, confusing_claim_img_encod, random_claim_img_encod, device):
    #     """
    #     In the forward function we accept a Tensor of input data and we must return
    #     a Tensor of output data. We can use Modules defined in the constructor as
    #     well as arbitrary operators on Tensors.
    #     """
    #     output_G = self.feed_forward(claim_img_encod=claim_img_encod,device=device)
    #     outpot_C = self.feed_forward(claim_img_encod=confusing_claim_img_encod,device=device)
    #     output_R = self.feed_forward(claim_img_encod=random_claim_img_encod,device=device)
    #     # return output_G
    #     return output_G, output_G - output_R, outpot_C - output_R
    #     return output_G, outpot_C, output_G - output_R, outpot_C - output_R
    
    # def inference(self, claim_img_encod,device):
    #     return self.feed_forward(claim_img_encod=claim_img_encod,device=device)

    def feed_forward(self,  claim_img_encod,device):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
         
        if 'pixel_values' in claim_img_encod.keys():
            has_image=True
            img_encod = claim_img_encod['pixel_values'].squeeze(0).to(device)
            claim_img_encod['pixel_values'] = img_encod
        else:
            has_image=False
        claim_ids = claim_img_encod['input_ids'].squeeze(0).to(device)
        claim_img_attention_masks = claim_img_encod['attention_mask'].squeeze(0).to(device)
        claim_img_encod['input_ids'] = claim_ids
        claim_img_encod['attention_mask'] = claim_img_attention_masks

        if self.is_project_embed =="y":
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_embeds
                img_evidences_embed=output_clip.image_embeds     #m, 512
                img_evidence_mask=None
            else:
                text_embeds = self.clipmodel.get_text_features(**claim_img_encod)
                img_evidences_embed=None
                img_evidence_mask=None
        else:
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 12,28,512
                img_evidences_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 1,50,768
                img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token    
                img_evidence_mask=torch.ones(img_evidences_embed.shape[0],img_evidences_embed.shape[1], dtype=img_evidences_embed.dtype, device=img_evidences_embed.device) 
            else:
                img_evidences_embed=None
                img_evidence_mask=None      
                text_model_output=self.clipmodel.text_model(**claim_img_encod) 
                text_embeds=text_model_output.last_hidden_state 
                text_embeds=text_embeds[:,1:]
                claim_img_attention_masks=claim_img_attention_masks[:,1:]

        claim_embed= text_embeds[0:1]        #1,sequence size-1, 512.  remove CLS token
        claim_mask= claim_img_attention_masks[0:1]
        txt_evidences_embed= text_embeds[1:]    #n,sequence size, 512
        txt_evidence_mask=claim_img_attention_masks[1:] #TODO remove last SEP token     
        output=self.verifier(self.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        return output
        # return output, has_image, claim_embed, txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask

    def data_processing(self,claim_img_encod,device):
        if 'pixel_values' in claim_img_encod.keys():
            has_image=True
            img_encod = claim_img_encod['pixel_values'].squeeze(0).to(device)
            claim_img_encod['pixel_values'] = img_encod
        else:
            has_image=False
        claim_ids = claim_img_encod['input_ids'].squeeze(0).to(device)
        claim_img_attention_masks = claim_img_encod['attention_mask'].squeeze(0).to(device)
        claim_img_encod['input_ids'] = claim_ids
        claim_img_encod['attention_mask'] = claim_img_attention_masks

        if self.is_project_embed =="y":
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_embeds
                img_evidences_embed=output_clip.image_embeds     #m, 512
                img_evidence_mask=None
            else:
                text_embeds = self.clipmodel.get_text_features(**claim_img_encod)
                img_evidences_embed=None
                img_evidence_mask=None
        else:
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 12,28,512
                img_evidences_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 1,50,768
                img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token    
                img_evidence_mask=torch.ones(img_evidences_embed.shape[0],img_evidences_embed.shape[1], dtype=img_evidences_embed.dtype, device=img_evidences_embed.device) 
            else:
                img_evidences_embed=None
                img_evidence_mask=None      
                text_model_output=self.clipmodel.text_model(**claim_img_encod) 
                text_embeds=text_model_output.last_hidden_state 
                text_embeds=text_embeds[:,1:]
                claim_img_attention_masks=claim_img_attention_masks[:,1:]

        claim_embed= text_embeds[0:1]        #1,sequence size-1, 512.  remove CLS token
        claim_mask= claim_img_attention_masks[0:1]
        txt_evidences_embed= text_embeds[1:]    #n,sequence size, 512
        txt_evidence_mask=claim_img_attention_masks[1:] #TODO remove last SEP token   
        return has_image, claim_embed, txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask

    def has_evidence(self,has_image):
        if not self.model_type_enum.use_text_evidence and (not  self.model_type_enum.use_image_evidence or not has_image):
            return False
        else:
            return True

# class 

class Verifier(torch.nn.Module):        
    def __init__(self, max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method ):    
        super(Verifier, self).__init__()
        self.only_claim_setting=only_claim_setting 
        if model_type_enum.stance_layer=="2":
            self.stance_detect_layer=StanceDetectionLayer2(512,max_image_num,model_type,model_type_enum)
        elif model_type_enum.stance_layer=="3":
            self.stance_detect_layer=StanceDetectionLayer3(512,max_image_num,model_type,model_type_enum)
        elif model_type_enum.stance_layer=="4": 
            self.stance_detect_layer= BertMultiwayMatch( 768,512,model_type_enum)
        elif model_type_enum.stance_layer=="5": 
            self.stance_detect_layer= BertMultiwayMatch2( 768,512,model_type_enum)
        elif model_type_enum.stance_layer=="6": 
            self.stance_detect_layer= BertMultiwayMatch3( 768,512,model_type_enum)    
        else:
            self.stance_detect_layer=StanceDetectionLayer(512,max_image_num,model_type,model_type_enum)
       
        if model_type_enum.sentiment_layer =="4":
            self.sentiment_detect_layer=SentimentDetectionLayer4(512,token_pooling_method)
        elif model_type_enum.sentiment_layer =="2":
            self.sentiment_detect_layer=SentimentDetectionLayer2(512,token_pooling_method)
        elif model_type_enum.sentiment_layer =="5":
            self.sentiment_detect_layer=SentimentDetectionLayer5(512,token_pooling_method)
        else:
            self.sentiment_detect_layer=SentimentDetectionLayer(512,token_pooling_method)
            
            
    def forward(self,   has_evidence,claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask,use_text=True):
        if has_evidence:
            output=self.stance_detect_layer(claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask, use_text)
        elif self.only_claim_setting=="stance":
            output=self.stance_detect_layer(claim_embed,claim_embed,None,False,claim_mask,claim_mask,None)
        else:
            output=self.sentiment_detect_layer(claim_embed  )
        return output